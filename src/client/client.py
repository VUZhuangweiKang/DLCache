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
import boto3
from collections import OrderedDict, defaultdict
import numpy as np
from pymongo import MongoClient
import concurrent
import multiprocessing
import datetime
import bson
from utils import *


logger = get_logger(__name__, level='Debug')

# data sharing channels between DLJob and Client
prefetchChannel = "/share/prefetchKeys.json"
dataReqChannel = "/share/next"
dataMissChannel = '/share/datamiss'


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

        
class Client(pyinotify.ProcessEvent):
    def __init__(self):
        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        # delete the old prefetchKeys
        if os.path.exists(prefetchChannel):
            os.remove(prefetchChannel)
        if os.path.exists(dataReqChannel):
            os.remove(dataReqChannel)
            
        # register the job to Manager
        if len(self.jobsmeta) > 0:
            self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
            self.channel = grpc.insecure_channel("dlcpod-manager:50051")
            self.conn_stub = pb_grpc.ConnectionStub(self.channel)
            
            req = pb.ConnectRequest(
                cred=self.cred, 
                s3auth=pb.S3Auth(**cloudSecret),
                createUser=True
            )
            resp = self.conn_stub.connect(req)
            if resp.rc == pb.RC.FAILED:
                logger.error("failed to connect to server with: {}".format(resp.resp))
                raise Exception
            else:
                logger.info("connect to server")
            
            self.register_stub = pb_grpc.RegistrationStub(self.channel)
            self.datamiss_stub = pb_grpc.DataMissStub(self.channel)
            self.dataset_col = None
            self.register_job()
        
        # ------------------------ Coordinated Data Prefetching ------------------------
        # create inotify
        wm = pyinotify.WatchManager()
        wm.add_watch("/share", pyinotify.IN_CLOSE_WRITE)
        
        # watch NFS data access events, which will be used for data eviction
        self.nfs_servers = os.popen(cmd="df -h | grep nfs | awk '{ print $6 }'").read().strip().split('\n')
        for svr in self.nfs_servers:
            if os.path.exists('/runtime{}'.format(svr)):
                wm.add_watch('/runtime{}'.format(svr), pyinotify.IN_CLOSE_NOWRITE)

        self.notifier = pyinotify.Notifier(wm, self)
        notifierThread = threading.Thread(target=self.notifier.loop, daemon=True)
        notifierThread.start()

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)      
                    
    def reset(self):        
        # cache is used to track which file is in tmpfs
        self.cache = defaultdict(set) # {batch_index: [tmpfs_path]}
        self.waterline = 2
        
        # record data requesting and loading time for dynamically tuning the cache capacity
        self.req_time = []
        self.load_time = []
        
        # runtime tmpfs waterline: n*num_workers, n=2 initially
        self.prefetchIndex = 0
        self.runtimeConf = None
        self.prefetchPaths = None
        
    def exit_gracefully(self):
        self.channel.close()
        self.notifier.stop()
            
    def register_job(self):
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
            resp = self.register_stub.register(request)
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
            
            if not self.dataset_col:
                mongo_client = MongoClient(resp.mongoUri)
                self.dataset_col = mongo_client.Cacher.Datasets

    def handle_datamiss(self):
        with open(dataMissChannel, 'r') as f:
            etags = f.readlines()
        for etag in etags:
            resp = self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag.strip('\n')))
            if resp.response:
                logger.info('request missing etag {}'.format(etag))
            else:
                logger.warning('failed to request missing etag {}'.format(etag))

    def prefetch(self):
        def docopy(nfs_path):
            tmpfs_path = '/runtime{}'.format(nfs_path)
            if nfs_path not in self.cache[self.prefetchIndex]:
                t = time.time()
                copyfile(nfs_path, tmpfs_path)  # NFS --> tmpfs
                self.load_time.append(time.time()-t)
                self.cache[self.prefetchIndex].add(tmpfs_path)

        for _ in range(self.runtimeConf['num_workers']):
            if self.prefetchIndex < len(self.prefetchPaths):
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    futures = []
                    for nfs_path in self.prefetchPaths[self.prefetchIndex]:
                        futures.append(executor.submit(docopy, nfs_path))
                    concurrent.futures.wait(futures)
                self.prefetchIndex += 1
                if self.prefetchIndex >= len(self.prefetchPaths):
                    self.prefetchIndex = 0
                    break

    def releaseCache(self, prefetchIndex):
        if prefetchIndex not in self.cache:
            return
        for tmpfspath in self.cache[prefetchIndex]:
            if os.path.exists(tmpfspath):
                os.remove(tmpfspath)
        del self.cache[prefetchIndex]
    
    def process_IN_CLOSE_WRITE(self, event):
        path = event.pathname
        print(path)
        if path == dataMissChannel:
            self.handle_datamiss()
        elif path == dataReqChannel:
            self.prefetch()
            self.req_time.append(time.time())
        elif path == prefetchChannel:
            self.reset()
            with open(prefetchChannel, 'r') as f:
                tmp = json.load(f)
                self.runtimeConf = tmp['meta']
                self.prefetchPaths = tmp['paths']
                
            # DLCJob is responsible for setting up the prefetchChannel
            # Client then set up the dataReqChannel, 
            # DLCJob is blocked until the dataReqChannel is created
            if not os.path.exists(dataReqChannel):
                # (waterline-1) because the below writting operation triggers the process_IN_CLOSE_WRITE
                for _ in range(self.waterline-1):
                    self.prefetch()
                    self.req_time.append(time.time())
                with open(dataReqChannel, 'w') as f:  # push signal into the dataReqChannel for initial data loading
                    f.write(str(time.time()))

    def process_IN_CLOSE_NOWRITE(self, event):
        # print(event.pathname)
        if len(self.cache) > 1:
            for index in list(self.cache.keys()):
                # update data access history
                for tmpfspath in self.cache[index]:
                    etag = tmpfspath.split('/')[-1]
                    # now = datetime.utcnow().timestamp()
                    now = datetime.datetime.now().timestamp()
                    self.dataset_col.update_one(
                        {"ETag": etag}, 
                        {
                            "$set": {"LastAccessTime": bson.timestamp.Timestamp(int(now), inc=1)},
                            "$inc": {"TotalAccessTime": 1}
                        })
                    
                # release tmpfs cache
                if index < self.prefetchIndex-self.waterline:
                    self.releaseCache(index)

            # tune buffer size
            if len(self.req_time) > 2:
                """
                To ensure the data is always available for DataLoader, the length of buffer should be:
                s >= 2*B, if alpha >= beta; otherwise,
                s >= (N-k)*(1-alpha/beta) 
                """
                alpha = np.mean(np.diff(self.req_time)[1:])
                beta = np.mean(np.array(self.load_time))
                N = len(self.prefetchPaths) - len(self.req_time)
                s = max(self.waterline, (1-alpha/beta)*N)
                
                # update waterline according to load/consume speed
                if self.waterline == s:
                    return
                else:
                    self.waterline = s
                    while len(self.cache) > s:
                        self.releaseCache(self.prefetchIndex)
                        self.prefetchIndex -= 1
                    while len(self.cache) < s:
                        self.prefetch()


if __name__ == '__main__':
    try:
        Client()
        while True:
            continue
    except KeyboardInterrupt:
        pass
        