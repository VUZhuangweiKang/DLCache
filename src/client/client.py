import grpc
import signal
import json
import time
import math
import glob
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import numpy as np
from pymongo import MongoClient
import concurrent
import multiprocessing
import shutil
import zmq
from utils import *


logger = get_logger(__name__, level='Debug')
cloudSecret = {
    "aws_access_key_id": read_secret('aws_access_key_id'),
    "aws_secret_access_key": read_secret('aws_secret_access_key'),
    "region_name": read_secret('region_name')   
}
manager_uri = "dlcpod-manager:50051"
init_channel = 'ipc:///share/init.ipc'
ipc_channel = 'ipc:///share/runtime.ipc'

COOL_DOWN_SEC = 600
class CHUNK_STATUS:
    PREPARE = 0
    ACTIVE = 1
    PENDING = 2
    COOL_DOWN = 3
    INACTIVE = 4


class Client(object):
    def __init__(self):
        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        channel = grpc.insecure_channel(manager_uri)
        # register the job to Manager
        if len(self.jobsmeta) > 0:
            self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
            
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
            
            stub = pb_grpc.RegistrationStub(channel)
            mongoUri = self.registerJob(stub)
            mongo_client = MongoClient(mongoUri, connect=False)
            self.dataset_col = mongo_client.Cacher.Datasets
            self.datamiss_stub = pb_grpc.DataMissStub(channel)
                
        context = zmq.Context()
        self.socket_rep = context.socket(zmq.REP)
        self.socket_rep.bind(init_channel)
        self.socket_sub = context.socket(zmq.SUB)
        self.socket_sub.bind(ipc_channel)
        for topic in [b'prefetch', b'dataMiss', b'releaseCache', b'expireCache']:
            self.socket_sub.setsockopt(zmq.SUBSCRIBE, topic)
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket_rep, zmq.POLLIN)
        self.poller.register(self.socket_sub, zmq.POLLIN)
        
        self.cool_down_proc = None
        self.processEvents()

    def reset(self):                
        # record data requesting and loading time for dynamically tuning the cache capacity
        self.req_time = []
        self.load_time = []
        
        self.prefetch_idx = 0
        self.nfs_paths = None
            
    def registerJob(self, stub):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            for node in job['nodesequence']:
                p = '/runtime/{}'.format(node)
                if not os.path.exists(p):
                    os.mkdir(p)
            
            train_keys = ds['keys']['train']
            val_keys = ds['keys']['validation']
            test_keys = ds['keys']['test']
            request = pb.RegisterRequest(
                cred=self.cred,
                datasource=pb.DataSource(
                    name=ds['name'], 
                    bucket=ds['bucket'], 
                    keys=pb.JobDatasets(
                        train=pb.Dataset(**train_keys),
                        validation=pb.Dataset(**val_keys),
                        test=pb.Dataset(**test_keys)
                    )
                ),
                nodesequence=job['nodesequence'],
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
                job['jobId'] = resp.jobId
                job['mongoUri'] = resp.mongoUri
                with open('/share/{}.json'.format(job['name']), 'w') as f:
                    json.dump(job, f)
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))
        return resp.mongoUri

    def prefetch(self, idx):
        def docopy(nfs_path):
            if os.path.exists(nfs_path):
                tmpfs_path = '/runtime{}'.format(nfs_path)
                t = time.time()
                shutil.copyfile(nfs_path, tmpfs_path, follow_symlinks=True)  # NFS --> tmpfs
                if os.stat(nfs_path).st_size == os.stat(tmpfs_path).st_size:
                    self.load_time.append(time.time()-t)
                    # print('copy file {}'.format(nfs_path))
                    assert os.path.exists(tmpfs_path)
                    while os.stat(nfs_path).st_size != os.stat(tmpfs_path).st_size: pass
                    return True, nfs_path
            # print('failed to copy {}'.format(nfs_path))
            return False, nfs_path

        def create_path(nfs_path):
            tmpfs_path = '/runtime{}'.format(nfs_path)
            root_folder = '/'.join(tmpfs_path.split('/')[:-1])
            if not os.path.exists(root_folder):
                os.makedirs(root_folder)
                        
        if idx < len(self.nfs_paths):
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for sample_path, target_path in self.nfs_paths[idx]:
                    create_path(sample_path)
                    futures.append(executor.submit(docopy, sample_path))
                    if target_path:
                        create_path(target_path)
                        futures.append(executor.submit(docopy, target_path))

            for task in futures:
                rc, miss_file = task.result()
                etag = miss_file.split('/')[-1]
                if not rc:
                    self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag))

    def expireChunks(self):
        time.sleep(COOL_DOWN_SEC)
        etags = []
        for sample_path, target_path in self.nfs_paths:
            etags.append(sample_path.split('/')[-1])
            if target_path:
                etags.append(target_path.split("/")[-1])
        self.dataset_col.update_many(
            {
                "ChunkETag": {"$in": etags},
                "Status.code": CHUNK_STATUS.COOL_DOWN
            },
            {
                "$set":{
                    "Status.code": CHUNK_STATUS.INACTIVE,
                    "Status.active_count": 0   
                }
            }
        )
    
    def processEvents(self):
        batch_size = None
        while True:
            socks = dict(self.poller.poll())
            if self.socket_rep in socks and socks[self.socket_rep] == zmq.POLLIN:
                topic, data = self.socket_rep.recv_multipart()
                topic, data = topic.decode("utf-8"), data.decode("utf-8")
                if topic == "init":
                    self.reset()
                    data = json.loads(data)                
                    batch_size = data["batch_size"]
                    self.nfs_paths = data['paths']
                    self.cache_size = data["num_workers"] * data["prefetch_factor"]                
                    for i in range(self.cache_size):
                        self.prefetch(i)
                    self.socket_rep.send(b'')

            if self.socket_sub in socks and socks[self.socket_sub] == zmq.POLLIN:
                topic, data = self.socket_sub.recv_multipart()
                topic, data = topic.decode("utf-8"), data.decode("utf-8")
                logger.info('recv msg: {} {}'.format(topic, data))
                if topic == "prefetch":
                    batch_idx = [int(idx) for idx in data.split(',') if len(idx) > 0]
                    for idx in batch_idx:
                        if idx >= self.prefetch_idx:
                            self.prefetch(idx)
                            self.prefetch_idx += 1
                            self.req_time.append(time.time())
                        else:
                            break
                    
                    # tune cache capacity
                    if len(self.req_time) > 3:
                        """
                        To ensure the data is always available for DataLoader, the length of buffer should be:
                        s >= 2*B, if alpha >= beta; otherwise,
                        s >= (N-k)*(1-alpha/beta)
                        """
                        reqIntervals = np.mean(np.diff(self.req_time)[1:])  # data consumption intervals
                        loadIntervals = np.mean(np.array(self.load_time))  # data loading intervals
                        remainingBatches = len(self.nfs_paths) - len(self.req_time)  # remaining batches
                        capacity = max(self.cache_size, math.ceil((1-reqIntervals/loadIntervals) * remainingBatches))
                        
                        # expand the cache capacity
                        for _ in range(self.cache_size, capacity+1):
                            self.prefetch(self.prefetch_idx)
                            self.prefetch_idx += 1
                        self.cache_size = capacity
                elif topic == "dataMiss":
                    miss_info, sub_idx = data.split(' ')
                    miss_idx, miss_etag = miss_info.split(":")
                    miss_idx, sub_idx = int(miss_idx), int(sub_idx)
                    self.nfs_paths[miss_idx//batch_size][miss_idx%batch_size], self.nfs_paths[sub_idx//batch_size][sub_idx%batch_size] = self.nfs_paths[sub_idx//batch_size][sub_idx%batch_size], self.nfs_paths[miss_idx//batch_size][miss_idx%batch_size]
                    self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=miss_etag))
                elif topic == "releaseCache":
                    batch_idx = int(data)
                    if batch_idx < len(self.nfs_paths):
                        def release(path):
                            if path:
                                tmpfspath = '/runtime' + path
                                if os.path.exists(tmpfspath):
                                    os.remove(tmpfspath)
                        for sample_path, target_path in self.nfs_paths[batch_idx]:            
                            release(sample_path)
                            release(target_path)
                elif topic == "expireCache":
                    if self.cool_down_proc is not None and self.cool_down_proc.is_alive():
                        self.cool_down_proc.terminate()
                    self.cool_down_proc = multiprocessing.Process(target=self.expireChunks, daemon=True)
                    self.cool_down_proc.start()


if __name__ == '__main__':
    client = Client()