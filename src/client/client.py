from __future__ import print_function
import grpc
import signal
import json
import time
import glob
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import pyinotify
import threading
from collections import OrderedDict, defaultdict
import numpy as np
from pymongo import MongoClient
import concurrent
import multiprocessing
import datetime
import bson
import zmq
from utils import *


logger = get_logger(__name__, level='Debug')

def read_secret(arg):
    path = '/secret/{}'.format(arg)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data


cloudSecret = {
    "aws_access_key_id": read_secret('aws_access_key_id'),
    "aws_secret_access_key": read_secret('aws_secret_access_key'),
    "region_name": read_secret('region_name')   
}
managerUri = "dlcpod-manager:50051"
initChannel = "/share/prefetchKeys.json"

        
class Client(object):
    def __init__(self):
        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        # delete the old prefetchKeys
        if os.path.exists(initChannel):
            os.remove(initChannel)
            
        # register the job to Manager
        if len(self.jobsmeta) > 0:
            self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
            with grpc.insecure_channel(managerUri) as channel:
                conn_stub = pb_grpc.ConnectionStub(channel)
                req = pb.ConnectRequest(
                    cred=self.cred, 
                    s3auth=pb.S3Auth(**cloudSecret),
                    createUser=True
                )
                resp = conn_stub.connect(req)
                if resp.rc == pb.RC.FAILED:
                    logger.error("failed to connect to server with: {}".format(resp.resp))
                    raise Exception
                else:
                    logger.info("connect to server")
            
            with grpc.insecure_channel(managerUri) as channel:
                stub = pb_grpc.RegistrationStub(channel)
                mongoUri = self.registerJob(stub)
                mongo_client = MongoClient(mongoUri, connect=False)
                self.dataset_col = mongo_client.Cacher.Datasets
                
        context = zmq.Context()
        self.socket = context.Socket(zmq.SUB)
        self.socket.connect('tcp://localhost:5555')
        for topic in [b'init', b'prefetch', b'dataMiss', b'releaseCache']:
            self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.processEvents()
                
    def reset(self):        
        # cache is used to track which file is in tmpfs
        self.cache = OrderedDict() # {batch_index: [tmpfs_path]}
        
        # record data requesting and loading time for dynamically tuning the cache capacity
        self.req_time = []
        self.load_time = []
        
        self.prefetchIndex = 0
        self.prefetchPaths = None
            
    def registerJob(self, stub):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['dataSource']
            if 'keys' not in ds: ds['keys'] = []
            request = pb.RegisterRequest(
                cred=self.cred,
                datasource=pb.DataSource(name=ds['name'], bucket=ds['bucket'], keys=ds['keys']),
                nodeSequence=job['nodeSequence'],
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))

            with open('/share/{}.json'.format(job['name']), 'w') as f:  # marshelled registration response
                json.dump(MessageToDict(resp), f)
        return resp.mongoUri

    def handleDataMiss(self, etag):
        with grpc.insecure_channel(managerUri) as channel:
            datamiss_stub = pb_grpc.DataMissStub(channel)
            resp = datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag.strip('\n')))
            if resp.response:
                logger.info('request missing etag {}'.format(etag))
            else:
                logger.warning('failed to request missing etag {}'.format(etag))

    def prefetch(self):
        def docopy(nfs_path):
            tmpfs_path = '/runtime{}'.format(nfs_path)
            t = time.time()
            copyfile(nfs_path, tmpfs_path)  # NFS --> tmpfs
            self.load_time.append(time.time()-t)

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for nfs_path in self.prefetchPaths[self.prefetchIndex]:
                futures.append(executor.submit(docopy, nfs_path))
            concurrent.futures.wait(futures)
        self.prefetchIndex += 1

    def processEvents(self):
        while True:
            topic, data = self.socket.recv()
            topic = topic.decode("utf-8")
            data = data.decode("utf-8")
            if topic == "init":
                self.reset()
                with open(initChannel, 'r') as f:
                    tmp = json.load(f)
                    self.prefetchPaths = tmp['paths']
                    self.minCacheCapacity = 2
            elif topic == "prefetch":
                batchIndex = int(data)
                # catch up the next batch
                while self.prefetchIndex <= batchIndex:
                    self.prefetch()
                    self.req_time.append(time.time())
                
                # tune cache capacity
                if len(self.req_time) > 3:
                    """
                    To ensure the data is always available for DataLoader, the length of buffer should be:
                    s >= 2*B, if alpha >= beta; otherwise,
                    s >= (N-k)*(1-alpha/beta) 
                    """
                    reqIntervals = np.mean(np.diff(self.req_time)[1:])  # data consumption intervals
                    loadIntervals = np.mean(np.array(self.load_time))  # data loading intervals
                    remainingBatches = len(self.prefetchPaths) - len(self.req_time)  # remaining batches
                    capacity = max(self.minCacheCapacity, (1-reqIntervals/loadIntervals) * remainingBatches)
                    
                    # expand the cache capacity
                    for _ in range(capacity-self.minCacheCapacity+1):
                        self.prefetch()
                    self.minCacheCapacity = capacity
            elif topic == "dataMiss":
                self.handleDataMiss(etag=data)
            elif topic == "releaseCache":
                files = data.split(",")
                for tmpfspath in files:
                    if not os.path.exists(tmpfspath): continue
                    etag = tmpfspath.split('/')[-1]
                    now = datetime.datetime.now().timestamp()
                    self.dataset_col.update_one(
                        {"ETag": etag}, 
                        {
                            "$set": {"LastAccessTime": bson.timestamp.Timestamp(int(now), inc=1)},
                            "$inc": {"TotalAccessTime": 1}
                        })
                    if os.path.exists(tmpfspath):
                        os.remove(tmpfspath)


if __name__ == '__main__':
    client = Client()