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
from datetime import datetime
from collections import defaultdict
import concurrent.futures
import bson
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
            resp = self.registerJob(stub)
            mongo_client = MongoClient(resp.mongoUri, connect=False)
            self.dataset_col = mongo_client.Cacher.Datasets
            self.job_col = mongo_client.Cacher.Job

            self.create_mappings(resp.jobId)
            # DL application is blocked until this file is writen
            with open('/share/{}.json'.format(job['name']), 'w') as f:
                json.dump(job, f)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))

            self.datamiss_stub = pb_grpc.DataMissStub(channel)
        
        context = zmq.Context()
        self.socket_rep = context.socket(zmq.REP)
        self.socket_rep.bind(init_channel)
        self.socket_sub = context.socket(zmq.SUB)
        self.socket_sub.bind(ipc_channel)
        for topic in [b'loadCache', b'releaseCache', b'expireChunk']:
            self.socket_sub.setsockopt(zmq.SUBSCRIBE, topic)
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket_rep, zmq.POLLIN)
        self.poller.register(self.socket_sub, zmq.POLLIN)
        
        self.cool_down_proc = None
        self.tmpfs_paths = defaultdict(list)
        
        self.load_cache_proc = None
        self.process_events()
        
    def registerJob(self, stub):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            
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
                return resp.regsucc
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            
    def create_mappings(self, jobId):
        """Create object mapings between cloud objects and NFS files
        """
        cursor = self.job_col.find_one({"Meta.JobId": jobId})
        job_info = {"ChunkETags": cursor["ChunkETags"], "Meta": cursor["Meta"]}
        
        def load(etags):
            data = {}
            if etags:
                chunks_iter = self.dataset_col.aggregate([
                    {"$match": {"ChunkETag": {"$in": etags}}},
                    {"$project": {"Key": 1, "Location": 1, "ChunkETag": 1, "_id": 0}}
                ])
                for chunk in chunks_iter:
                    cloud_path, loc, chunk_etag = chunk['Key'], chunk["Location"], chunk['ChunkETag']
                    local_path = '/runtime/{}/{}'.format(loc, chunk_etag)
                    # decompressed folder, so key has the .tar.gz extension
                    if os.path.isdir(local_path):
                        # We assume data won't be immediately deleted after being downloaded by Manager.
                        for root, dirs, files in os.walk(local_path):
                            for name in files:
                                p = os.path.join(root, name)
                                dummy_cloud_path = cloud_path + p.replace(local_path, '')
                                data[dummy_cloud_path] = p
                    else:
                        data[cloud_path] = local_path
            return data

        for dataset_type in job_info['ChunkETags']:
            chunk_etags = job_info['ChunkETags'][dataset_type]
            samples_manifest = load(chunk_etags['samples'])
            targets_manifest = load(chunk_etags['targets'])
            with open('/share/{}_samples_manifests.json'.format(dataset_type), 'w') as f:
                json.dump(samples_manifest, f)
            if len(targets_manifest) > 0:
                with open('/share/{}_targets_manifests.json'.format(dataset_type), 'w') as f:
                    json.dump(targets_manifest)
    
    def async_mongo_opt(self, mongo_opt_queue):
        while True:
            if len(mongo_opt_queue) == 0: continue
            func, args = mongo_opt_queue.pop()
            if func == 'update_many':
                self.dataset_col.update_many(*args)
                
    def load_cache(self, send_idx_queue, load_time):
        while True:
            if len(send_idx_queue) == 0: continue
            dataset_type, idx, start_from = send_idx_queue.pop(0)
            logger.info('load cache for batch {}'.format((dataset_type, idx, start_from)))
            t = time.time()
                
            # update chunk status to ACTIVE
            chunk_etags = []
            for sample_path, target_path in self.tmpfs_paths[dataset_type][idx]:
                chunk_etags.append(sample_path.split('/')[-1])
                if target_path:
                    chunk_etags.append(target_path.split('/')[-1])
                    
            now = datetime.utcnow().timestamp()
            self.mongo_opt_queue.append(('update_many', [{"ChunkETag": {"$in": chunk_etags}}, 
                {
                        "$set": { "Status.code": CHUNK_STATUS.ACTIVE},
                        "$inc": {"Status.active_count": 1},
                        "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}
                }]))
            
            if idx < len(self.tmpfs_paths[dataset_type]):
                def docopy(tmpfs_path):
                    nfs_path = tmpfs_path.replace('/runtime', '')
                    if os.path.exists(nfs_path):
                        root_folder = '/'.join(nfs_path.split('/')[:-1])
                        if not os.path.exists(root_folder):
                            os.makedirs(root_folder)
                        if not os.path.exists(tmpfs_path):
                            shutil.copyfile(nfs_path, tmpfs_path)
                        return
                    print('failed to copy {}'.format(nfs_path))
                    etag = nfs_path.split('/')[-1]
                    self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag))

                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    futures = []
                    for sample_path, target_path in self.tmpfs_paths[dataset_type][idx][start_from:]:
                        futures.append(executor.submit(docopy, sample_path))
                        if target_path is not None:
                            futures.append(executor.submit(docopy, target_path))
                    concurrent.futures.wait(futures)
                load_time.append(time.time()-t)
            
            del chunk_etags, dataset_type, idx, start_from

    def release_cache(self, rcvd_idx_queue):
        while True:
            if len(rcvd_idx_queue) == 0: continue
            dataset_type, idx = rcvd_idx_queue.pop()
            if idx < len(self.tmpfs_paths[dataset_type]):
                def release(path):
                    if os.path.exists(path):
                        os.remove(path)
                
                futures = []
                for sample_path, target_path in self.tmpfs_paths[dataset_type][idx]:            
                    with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
                        futures.append(executor.submit(release, sample_path))
                        if target_path:
                            futures.append(executor.submit(release, target_path))
                        concurrent.futures.wait(futures)
                logger.info('release cache for batch {}'.format(idx))
                                    
    def expire_chunks(self, dataset_type):
        time.sleep(COOL_DOWN_SEC)
        etags = []
        for idx in self.tmpfs_paths[dataset_type]:
            for sample_path, target_path in self.tmpfs_paths[dataset_type][idx]:
                etags.append(sample_path.split('/')[-1])
                if target_path:
                    etags.append(target_path.split("/")[-1])
        
        now = datetime.utcnow().timestamp()
        self.mongo_opt_queue.append(('update_many', [{
                "ChunkETag": {"$in": etags},
                "Status.active_count": 1
            },
            {"$set": {
                "Status.code": CHUNK_STATUS.INACTIVE,
                "Status.active_count": 0,
                "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
            }]))
        self.mongo_opt_queue.append(('update_many', [{
                "ChunkETag": {"$in": etags},
                "Status.active_count": {"$gt": 1}
            },
            {
                "$inc": {"Status.active_count": -1},
                "$set": {
                    "Status.code": CHUNK_STATUS.INACTIVE,
                    "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)
                }
            }]))
    
    def process_events(self):
        prefetch_factor = None
        batch_size = None
        # clear runtime cache
        for path in glob.glob('/runtime/*'):
            shutil.rmtree(path)
            os.mkdir(path)
        while True:
            socks = dict(self.poller.poll())
            if self.socket_rep in socks and socks[self.socket_rep] == zmq.POLLIN:
                topic, dataset_type, data = self.socket_rep.recv_multipart()
                topic, dataset_type, data = topic.decode("utf-8"), dataset_type.decode('utf-8'), data.decode("utf-8")
                if topic == "init":    
                    data = json.loads(data)
                    prefetch_factor = data['prefetch_factor']
                    
                    with open('/share/{}_samples_manifests.json'.format(dataset_type), 'r') as f:
                        samples_tmpfs_paths = np.array(list(json.load(f).values()))
                    targets_tmpfs_paths = None
                    if os.path.exists('/share/{}_targets_manifests.json'.format(dataset_type)):
                        with open('/share/{}_targets_manifests.json'.format(dataset_type), 'r') as f:
                            targets_tmpfs_paths = np.array(list(json.load(f).values()))
                        
                    batched_nfs_paths = []
                    for batch in data['paths']:
                        batched_nfs_paths.append(list(zip(samples_tmpfs_paths[batch], \
                            targets_tmpfs_paths[batch] if targets_tmpfs_paths else [None]*len(batch))))
                    self.tmpfs_paths[dataset_type] = batched_nfs_paths
                    
                    if self.load_cache_proc is not None:
                        while len(self.send_idx_queue) > 0 and self.send_idx_queue[0][1] > 8:
                            self.send_idx_queue.pop(0)
                            self.load_time.pop(0)
                        
                        while self.load_cache_proc.is_alive():
                            self.load_cache_proc.terminate()
                        else:
                            self.load_cache_proc.close()
                            
                        while self.release_cache_proc.is_alive():
                            self.release_cache_proc.terminate()
                        else:
                            self.release_cache_proc.close()
                        
                        while self.mongo_opt_proc.is_alive():
                            self.mongo_opt_proc.terminate()
                        else:
                            self.mongo_opt_proc.close()
                        del self.load_cache_proc, self.release_cache_proc, self.mongo_opt_proc
                    
                    manager = multiprocessing.Manager()
                    self.load_time = manager.list()
                    self.send_idx_queue = manager.list()
                    self.rcvd_idx_queue = manager.list()
                    self.mongo_opt_queue = manager.list()
                    
                    self.load_cache_proc = multiprocessing.Process(target=self.load_cache, args=(self.send_idx_queue, self.load_time), daemon=True)
                    self.release_cache_proc = multiprocessing.Process(target=self.release_cache, args=(self.rcvd_idx_queue,), daemon=True)
                    self.mongo_opt_proc = multiprocessing.Process(target=self.async_mongo_opt, args=(self.mongo_opt_queue,), daemon=True)
                    
                    self.load_cache_proc.start()
                    self.release_cache_proc.start()
                    self.mongo_opt_proc.start()
                    
                    self.socket_rep.send(b'')
            if self.socket_sub in socks and socks[self.socket_sub] == zmq.POLLIN:
                topic, dataset_type, data = self.socket_sub.recv_multipart()
                topic, dataset_type, data = topic.decode("utf-8"), dataset_type.decode('utf-8'), data.decode("utf-8")
                # logger.info('recv msg: {} {}'.format(topic, data))
                if topic == "loadCache":
                    data = json.loads(data)
                    '''
                    Dataloader spends fetch_time / active_workers time to fetch 1 batch, 
                    while client spends load_time to move 1 batch from NFS to tmpfs.
                    1. If load_time < fetch_time / active_workers, client only need to push 1 batch into memory 
                        when it recieve the loadCache messgae.
                    2. Otherwise, cache missing may occur. To mitigate the situation in the remaining batches, 
                        the client can start loading cache from the position of k samples behind the starting position of each batch.
                        k = (active_workers * load_time) / fetch_time
                    Overall, we can say k = floor((active_workers * load_time) / fetch_time)
                    '''
                    if data['rcvd_idx'] == len(self.tmpfs_paths[dataset_type]):
                        continue
                    batch_size = len(self.tmpfs_paths[dataset_type][0])
                    # clean up pending batches, and prepare to load the next epoch
                    if data['send_idx'] == len(self.tmpfs_paths[dataset_type]):
                        if data['send_idx'] - data['rcvd_idx'] == data['active_workers'] * prefetch_factor:
                            try:
                                while len(self.send_idx_queue) > 0:
                                    item = self.send_idx_queue.pop(0)
                                    logger.info('pop item from queue: {}'.format(item))
                            except:
                                pass
                        data['send_idx'] = (data['rcvd_idx'] + data['active_workers'] * prefetch_factor) % len(self.tmpfs_paths[dataset_type])

                    send_idx = data['send_idx']
                    if np.isnan(data['fetch_time']) or len(self.load_time) == 0:
                        k = 0
                    else:
                        avg_load_time = np.mean(self.load_time)
                        # k = avg_load_time * data['active_workers'] / data['fetch_time']
                        k = avg_load_time / data['fetch_time']
                        k = math.ceil(batch_size * (k-1) / k) if k > 1 else 0
                    
                    self.send_idx_queue.insert(0, (dataset_type, send_idx, k))
                elif topic == "releaseCache":
                    idx = int(data)
                    self.rcvd_idx_queue.append((dataset_type, idx))
                elif topic == "expireChunk":
                    if self.cool_down_proc is not None and self.cool_down_proc.is_alive():
                        self.cool_down_proc.terminate()
                    self.cool_down_proc = multiprocessing.Process(target=self.expire_chunks, args=(dataset_type,), daemon=True)
                    self.cool_down_proc.start()


if __name__ == '__main__':
    client = Client()