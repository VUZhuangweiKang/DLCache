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
from collections import OrderedDict
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
                s3auth=pb.S3Auth(
                    aws_access_key_id=read_secret('aws_access_key_id'),
                    aws_secret_access_key=read_secret('aws_secret_access_key'),
                    region_name=read_secret('region_name')
                ),
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
        # runtimeBuffer is used to track which file is consumed in tmpfs
        self.runtimeBuffer = OrderedDict() # {batch_index: [tmpfs_path]}
        
        # record data requesting and loading time for dynamically tuning the cache capacity
        self.req_time = []
        self.load_time = []
        
        # runtime tmpfs waterline: n*num_workers*batch_size, n=2 initially
        self.waterline = 2
        self.prefetchIndex = 0
        self.runtimeConf = None
        self.prefetchPaths = None
        
        # create inotify
        wm = pyinotify.WatchManager()
        mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_CLOSE_NOWRITE
        wm.add_watch("/share", pyinotify.IN_CLOSE_WRITE, auto_add=True, rec=True)
        wm.add_watch('/runtime', pyinotify.IN_CLOSE_NOWRITE, auto_add=True, rec=True)
        
        # watch NFS data access events, which will be used for data eviction
        self.nfs_servers = os.popen(cmd="df -h | grep nfs | awk '{ print $6 }'").read().strip().split('\n')
        for svr in self.nfs_servers:
            wm.add_watch(svr, mask, auto_add=True, rec=True)

        self.notifier = pyinotify.Notifier(wm, self)
        notifierThread = threading.Thread(target=self.notifier.loop, daemon=True)
        notifierThread.start()
        
        # prefetch the first 2 batches
        while not self.prefetchPaths: continue
        for _ in range(self.waterline):
            with open(dataReqChannel, 'w') as f:  # push signal into the dataReqChannel for initial data loading
                f.write(str(time.time()))
        
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
            
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
            resp = self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag))
            if resp.response:
                logger.info('request missing etag {}'.format(etag))
            else:
                logger.warning('failed to request missing etag {}'.format(etag))

    def prefetch(self):
        def docopy(path):
            if self.prefetchIndex not in self.runtimeBuffer:
                self.runtimeBuffer[self.prefetchIndex] = []
            if path not in self.runtimeBuffer[self.prefetchIndex]:
                t = time.time()
                tmpfs_path = '/runtime{}'.format(path)
                copyfile(path, tmpfs_path)  # NFS --> tmpfs
                self.load_time.append(time.time()-t)
                self.runtimeBuffer[self.prefetchIndex].append(tmpfs_path)
                            
        if self.runtimeConf['LazyLoading']:
            for _ in range(self.runtimeConf['num_workers']):
                if self.prefetchIndex < len(self.prefetchPaths):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        futures = []
                        for path in self.prefetchPaths[self.prefetchIndex]:
                            futures.append(executor.submit(docopy, path))
                        concurrent.futures.wait(futures)
                    self.prefetchIndex += 1
                    if self.prefetchIndex >= len(self.prefetchPaths):
                        self.prefetchIndex = 0
                        break
        else:
            path = self.prefetchPaths[self.prefetchIndex]
            t = time.time()
            copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
            self.load_time.append(time.time()-t)
            self.prefetchIndex += 1
            if self.prefetchIndex == len(self.prefetchPaths):
                self.prefetchIndex = 0

    def process_IN_CLOSE_WRITE(self, event):
        path = event.pathname
        if path == dataMissChannel:
            return self.handle_datamiss()
        elif path == dataReqChannel:
            self.prefetch()
            self.req_time.append(time.time())
        elif path == prefetchChannel:
            with open(prefetchChannel, 'r') as f:
                tmp = json.load(f)
                self.runtimeConf = tmp['meta']
                self.prefetchPaths = tmp['paths']

    # TODO: bug: prefetch copy 文件的速度跟不上DLCJob请求数据的速度，所以要么出现data miss，要么碎片化的数据被load，造成读取失败
    def process_IN_CLOSE_NOWRITE(self, event):
        path = event.pathname.strip().split('/')[1:]
        if 'runtime' in path and len(path) > 2:
            if len(self.runtimeBuffer) > 1:
                # LRU Eviction
                prefetchIndex = list(self.runtimeBuffer.keys())[0]
                if len(self.runtimeBuffer[prefetchIndex]) > 0:
                    self.runtimeBuffer[prefetchIndex].pop(0)
                    if len(self.runtimeBuffer[prefetchIndex]) == 0:
                        self.runtimeBuffer.popitem(last=False)
            
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
                    s = max(2, (1-alpha/beta)*N)
                    
                    # update waterline according to load/consume speed
                    if self.waterline == s:
                        return
                    else:
                        self.waterline = s
                        while len(self.runtimeBuffer) > s:
                            self.runtimeBuffer.popitem(last=True)
                            self.prefetchIndex -= 1
                        while len(self.runtimeBuffer) < s:
                            self.prefetch()
        elif path[0] in self.nfs_servers:
            assert self.dataset_col is not None
            etag = path[1]
            now = datetime.utcnow().timestamp()
            self.dataset_col.update_one(
                {"ETag": etag}, 
                {
                    "$set": {"LastAccessTime": bson.timestamp.Timestamp(int(now), inc=1)},
                    "$inc": {"TotalAccessTime": 1}
                })


if __name__ == '__main__':
    try:
        Client()
        while True:
            pass
    except KeyboardInterrupt:
        pass
        