from __future__ import print_function
import grpc
import signal
import json
import time
import glob
from pathlib import Path
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import pyinotify
import shutil
import threading
from collections import OrderedDict
import numpy as np
from pymongo import MongoClient
import datetime
import bson
from utils import *


logger = get_logger(__name__, level='Debug')


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
        
        # runtimeBuffer is used to track which file is consumed in tmpfs
        self.runtimeBuffer = OrderedDict() # {batch_index: [tmpfs_path]}
        self.req_time = []
        self.load_time = []
        
        # runtime tmpfs waterline: n*num_workers*batch_size, n=2 initially
        self.waterline = 2
        self.prefetchIndex = 0
        # prefetchPaths is the file list in nfs
        try:
            with open('/share/prefetchKeys.json', 'r') as f:
                tmp = json.load(f)
                self.runtime_conf = tmp['meta']
                self.prefetchPaths = tmp['policy']
        except FIleNotFoundError:
            self.prefetchPaths = None
        
        # create inotify
        wm = pyinotify.WatchManager()
        mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_NOWRITE
        wm.add_watch("/share", mask, auto_add=True, rec=True)
        wm.add_watch('/runtime', mask, auto_add=True, rec=True)
        
        # watch data access events
        self.nfs_servers = os.popen(cmd="df -h | grep nfs | awk '{ print $6 }'").read().strip().split('\n')
        for svr in self.nfs_servers:
            wm.add_watch(svr, mask, auto_add=True, rec=True)

        self.notifier = pyinotify.Notifier(wm, self)
        notifierThread = threading.Thread(target=self.notifier.loop, daemon=True)
        notifierThread.start()
        
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
            self.register_job()

            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
        
        self.dataset_col = None
            
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
            
            mongo_client = MongoClient(resp.mongoUri)
            self.dataset_col = mongo_client.Cacher.Datasets

            # prefetch the first 2 batches
            while not self.prefetchPaths: pass
            for _ in range(self.waterline): 
                self.prefetch()

    def handle_datamiss(self):
        with open('/share/datamiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            bucket, key = key.split(':')
            resp = self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, bucket=bucket, key=key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))

    def prefetch(self):
        if self.runtime_conf['LazyLoading']:
            for _ in range(self.runtime_conf['num_workers']):
                if self.prefetchIndex < len(self.prefetchPaths):
                    for path in self.prefetchPaths[self.prefetchIndex]:
                        if self.prefetchIndex not in self.runtimeBuffer:
                            self.runtimeBuffer[self.prefetchIndex] = []
                        if path not in self.runtimeBuffer[self.prefetchIndex]:
                            t = time.time()
                            tmpfs_path = '/runtime{}'.format(path)
                            copyfile(path, tmpfs_path)  # NFS --> tmpfs
                            self.load_time.append(time.time()-t)
                            self.runtimeBuffer[self.prefetchIndex].append(tmpfs_path)
                    self.prefetchIndex += 1
        else:
            path = self.prefetchPaths[self.prefetchIndex]
            t = time.time()
            copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
            self.load_time.append(time.time()-t)
            self.prefetchIndex += 1

    def process_IN_CREATE(self, event):
        if event.pathname == '/share/datamiss':
            return self.handle_datamiss()
        elif event.pathname == '/share/next':
            self.req_time.append(time.time())
            self.prefetch()
        elif event.path == '/share/prefetchKeys.json':
            with open('/share/prefetchKeys.json', 'r') as f:
                tmp = json.load(f)
                self.runtime_conf = tmp['meta']
                self.prefetchPaths = tmp['policy']

    def process_IN_MODIFY(self, event):
        self.process_IN_CREATE(event)

    # bug: tmpfs中的文件在读取前（中）被删除
    def process_IN_CLOSE_NOWRITE(self, event):
        path = event.pathname.strip().split('/')[1:]
        if 'runtime' in path and len(path) > 2:
            if len(self.runtimeBuffer) > 1:
                prefetchIndex = list(self.runtimeBuffer.keys())[0]
                if len(self.runtimeBuffer[prefetchIndex]) > 0:
                    tmpfsPath = self.runtimeBuffer[prefetchIndex].pop(0)
                    if len(self.runtimeBuffer[prefetchIndex]) == 0:
                        self.runtimeBuffer.popitem(last=False)
                    os.remove(tmpfsPath)
            
                # tune buffer size
                if len(self.req_time) > 2:
                    """
                    To ensure the data is always available for DataLoader, the length of buffer should be:
                    s >= 2*B, if alpha >= beta; otherwise,
                    s >= (N-k)*(1-alpha/beta) 
                    """
                    alpha = np.mean(np.diff(self.req_time[-4:])[1:])
                    beta = np.mean(np.array(self.load_time[-3:]))
                    N = len(self.prefetchPaths)
                    k = len(self.req_time)
                    s = max(2, (1-alpha/beta)*(N-k))
                    
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
        