
import time
import math
import glob
import shutil
import zmq
import pickle
import gc
import signal
import json, bson
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
from queue import Empty
import grpc
import threading
import databus.dbus_pb2 as pb
import databus.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
from utils import *

logger = get_logger(__name__, level='Debug')
cloudSecret = {
    "aws_access_key_id": read_secret('aws_access_key_id'),
    "aws_secret_access_key": read_secret('aws_secret_access_key'),
    "region_name": read_secret('region_name')   
}
cpu_count = mp.cpu_count()
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
            
            self.manager_stub = pb_grpc.ManagerStub(channel)
            req = pb.ConnectRequest(
                cred=self.cred, 
                s3auth=pb.S3Auth(**cloudSecret),
                createUser=True
            )
            resp = self.manager_stub.connect(req)
            if resp.rc == pb.RC.FAILED:
                logger.error("failed to connect to server with: {}".format(resp.resp))
                raise Exception
            else:
                logger.info("connect to server")
            
            resp = self.registerJob()
            mongo_client = MongoClient(resp.mongoUri, connect=True)
            self.dataset_col = mongo_client.Cacher.Datasets
            self.job_col = mongo_client.Cacher.Job

            self.create_mappings(resp.jobId)
            # DL application is blocked until this file is writen
            path = '/share/{}.json'.format(job['name'])
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump(job, f)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))
        
        context = zmq.Context()
        self.socket_rep = context.socket(zmq.REP)
        self.socket_rep.bind(init_channel)
        self.socket_sub = context.socket(zmq.SUB)
        self.socket_sub.connect(ipc_channel)
        for topic in [b'loadCache', b'releaseCache', b'expireChunk', b'stopIteration', b'missETags']:
            self.socket_sub.setsockopt(zmq.SUBSCRIBE, topic)
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket_rep, zmq.POLLIN)
        self.poller.register(self.socket_sub, zmq.POLLIN)
        
        self.tmpfs_paths = defaultdict(list)
        self.cool_down_proc = None
        self.load_cache_proc = None
        self.process_events()
        
    def registerJob(self):
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
            resp = self.manager_stub.register(request)
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
                    nfs_path = '/{}/{}'.format(loc, chunk_etag)
                    cache_path = '/runtime{}'.format(nfs_path)
                    # decompressed folder
                    if os.path.isdir(nfs_path):
                        # We assume data won't be immediately deleted after being downloaded by Manager.
                        for root, dirs, files in os.walk(nfs_path):
                            for name in files:
                                p = os.path.join(root, name)
                                dummy_cloud_path = cloud_path + p.replace(nfs_path, '')
                                data[dummy_cloud_path] = '/runtime{}'.format(p)
                    else:
                        data[cloud_path] = cache_path
            return data

        for dataset_type in job_info['ChunkETags']:
            chunk_etags = job_info['ChunkETags'][dataset_type]
            samples_manifest = load(chunk_etags['samples'])
            targets_manifest = load(chunk_etags['targets'])
            path = '/share/{}_samples_manifests.pkl'.format(dataset_type)
            if not os.path.exists(path):
                with open(path, 'wb') as f:
                    pickle.dump(samples_manifest, f)
            path = '/share/{}_targets_manifests.pkl'.format(dataset_type)
            if not os.path.exists(path) and len(targets_manifest) > 0:
                with open('/share/{}_targets_manifests.pkl'.format(dataset_type), 'wb') as f:
                    pickle.dump(targets_manifest)
    
    def async_mongo_opt(self, mongo_opt_queue):
        while True:
            func, args = mongo_opt_queue.get()
            if func == 'update_many':
                self.dataset_col.update_many(*args)
                
    def load_cache(self, send_idx_queue, tmpfs_paths):
        event = threading.Event()
        
        def docopy(items):
            for tmpfs_path in items:
                if not event.is_set():
                    break
                nfs_path = tmpfs_path.replace('/runtime', '')
                if os.path.exists(nfs_path):
                    while True:
                        try:
                            shutil.copyfile(nfs_path, tmpfs_path)
                            break
                        except Exception as ex:
                            root_folder = '/'.join(tmpfs_path.split('/')[:-1])
                            os.makedirs(root_folder, exist_ok=True)
                else:
                    print('failed to copy {}'.format(nfs_path))
                    etag = nfs_path.split('/')[-1]
                    self.manager_stub.handle_datamiss(pb.DataMissRequest(cred=self.cred, etag=etag))
        
        thrd = None
        while True:
            dataset_type, idx = send_idx_queue.get()
            if event.is_set():
                event.clear()
            
            # logger.info('load cache for batch {}, remaining {}'.format((dataset_type, idx, start_from), send_idx_queue.qsize()))
            if idx < len(tmpfs_paths[dataset_type]):
                items = []
                for sample_path, target_path in tmpfs_paths[dataset_type][idx][::-1]:
                    items.append(sample_path)
                    if target_path is not None:
                        items.append(target_path)
                
                while thrd is not None and thrd.is_alive():
                    pass
                else:
                    event.set()
                    thrd = threading.Thread(target=docopy, args=(items, ), daemon=True)
                    thrd.start()
            
            # update chunk status
            chunk_etags = []
            for sample_path, target_path in tmpfs_paths[dataset_type][idx]:
                chunk_etags.append(sample_path.split('/')[3])
                if target_path:
                    chunk_etags.append(target_path.split('/')[3])
            chunk_etags = list(set(chunk_etags))
            now = datetime.utcnow().timestamp()
            self.mongo_opt_queue.put(('update_many', [{"ChunkETag": {"$in": chunk_etags}}, 
                {
                        "$set": { "Status.code": CHUNK_STATUS.ACTIVE},
                        "$inc": {"Status.active_count": 1},
                        "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}
                }]))
            del chunk_etags

    def release_cache(self, rcvd_idx_queue, cache_usage, tmpfs_paths):
        while True:
            dataset_type, idx = rcvd_idx_queue.get()
            cache_usage.append(count_files('/runtime'))
            if idx < len(tmpfs_paths[dataset_type]):
                def release(path):
                    try:
                        os.remove(path)
                    except Exception as ex:
                        pass
                
                # futures = []
                # for sample_path, target_path in self.tmpfs_paths[dataset_type][idx]:            
                #     with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                #         futures.append(executor.submit(release, sample_path))
                #         if target_path:
                #             futures.append(executor.submit(release, target_path))
                #         concurrent.futures.wait(futures)
                
                for sample_path, target_path in tmpfs_paths[dataset_type][idx]:   
                    release(sample_path)
                    if target_path:
                        release(target_path)
                        
                # logger.info('release cache for batch {}'.format(idx))
                                    
    def expire_chunks(self, dataset_type, tmpfs_paths):
        time.sleep(COOL_DOWN_SEC)
        etags = []
        for idx in tmpfs_paths[dataset_type]:
            for sample_path, target_path in tmpfs_paths[dataset_type][idx]:
                etags.append(sample_path.split('/')[-1])
                if target_path:
                    etags.append(target_path.split("/")[-1])
        
        now = datetime.utcnow().timestamp()
        self.mongo_opt_queue.put(('update_many', [{
                "ChunkETag": {"$in": etags},
                "Status.active_count": 1
            },
            {"$set": {
                "Status.code": CHUNK_STATUS.INACTIVE,
                "Status.active_count": 0,
                "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
            }]))
        self.mongo_opt_queue.put(('update_many', [{
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
    
    @staticmethod
    def clear_runtime():
        for root, dirs, files in os.walk('/runtime', topdown=False):
            while True:
                try:
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        shutil.rmtree(os.path.join(root, name))
                    break
                except:
                    pass
                            
    def process_events(self):
        prefetch_factor = None
        batch_size = None
        window_size = None
        # clear runtime cache
        self.clear_runtime()
        
        while True:
            socks = dict(self.poller.poll())
            if self.socket_rep in socks and socks[self.socket_rep] == zmq.POLLIN:
                topic, dataset_type, data = self.socket_rep.recv_multipart()
                topic, dataset_type = topic.decode("utf-8"), dataset_type.decode('utf-8')
                if topic == "init":
                    self.socket_rep.send(b'')
                    data = pickle.loads(data)
                    prefetch_factor = data['prefetch_factor']
                    num_workers = data['active_workers']
                    window_size = num_workers * prefetch_factor
                    with open('/share/{}_samples_manifests.pkl'.format(dataset_type), 'rb') as f:
                        samples_tmpfs_paths = np.array(list(pickle.load(f).values()))
                    targets_tmpfs_paths = None
                    if os.path.exists('/share/{}_targets_manifests.pkl'.format(dataset_type)):
                        with open('/share/{}_targets_manifests.pkl'.format(dataset_type), 'rb') as f:
                            targets_tmpfs_paths = np.array(list(json.load(f).values()))

                    batched_tmpfs_paths = []
                    for batch in data['paths']:
                        batched_tmpfs_paths.append(list(zip(samples_tmpfs_paths[batch], \
                            targets_tmpfs_paths[batch] if targets_tmpfs_paths else [None]*len(batch))))
                    self.tmpfs_paths[dataset_type] = batched_tmpfs_paths
                    
                    self.send_idx_queue = mp.Queue()
                    self.send_idx_queue.cancel_join_thread()
                    self.rcvd_idx_queue = mp.Queue()
                    self.rcvd_idx_queue.cancel_join_thread()
                    self.mongo_opt_queue = mp.Queue()
                    self.mongo_opt_queue.cancel_join_thread()
                    
                    self.mp_manager = mp.Manager()
                    self.cache_usage = self.mp_manager.list()

                    self.load_cache_proc = mp.Process(target=self.load_cache, args=(self.send_idx_queue, self.tmpfs_paths), daemon=True)
                    self.release_cache_proc = mp.Process(target=self.release_cache, args=(self.rcvd_idx_queue, self.cache_usage, self.tmpfs_paths), daemon=True)
                    self.mongo_opt_proc = mp.Process(target=self.async_mongo_opt, args=(self.mongo_opt_queue,), daemon=True)
                    self.load_cache_proc.start()
                    self.release_cache_proc.start()
                    self.mongo_opt_proc.start()
                    
                    del batched_tmpfs_paths, targets_tmpfs_paths, samples_tmpfs_paths
        
            if self.socket_sub in socks and socks[self.socket_sub] == zmq.POLLIN:
                topic, dataset_type, data = self.socket_sub.recv_multipart()
                topic, dataset_type = topic.decode("utf-8"), dataset_type.decode('utf-8')
                # logger.info('recv msg: {} {}'.format(topic, data))
                if topic == "loadCache":
                    data = pickle.loads(data)
                    window_size = data['active_workers'] * prefetch_factor
                    if data['rcvd_idx'] == len(self.tmpfs_paths[dataset_type]):
                        continue
                    # clean up pending batches, and prepare to load the next epoch
                    if data['send_idx'] == len(self.tmpfs_paths[dataset_type]):
                        if data['send_idx'] - data['rcvd_idx'] == window_size:
                            while True:
                                try:
                                    item = self.send_idx_queue.get_nowait()
                                    logger.info('pop item from queue: {}'.format(item))
                                except:
                                    break
                        data['send_idx'] = (data['rcvd_idx'] + window_size) % len(self.tmpfs_paths[dataset_type])

                    send_idx = data['send_idx']
                    
                    '''
                    Due to measurement error, there might be backlog in send_idx_queue.
                    We skip those indexes to avoid cascading backlog.
                    '''
                    while True:
                        try:
                            self.send_idx_queue.get_nowait()
                        except:
                            break
                    self.send_idx_queue.put((dataset_type, send_idx))
                elif topic == "releaseCache":
                    idx = int(data)
                    self.rcvd_idx_queue.put((dataset_type, idx))
                elif topic == "expireChunk":
                    if self.cool_down_proc is not None and self.cool_down_proc.is_alive():
                        self.cool_down_proc.terminate()
                    self.cool_down_proc = mp.Process(target=self.expire_chunks, args=(dataset_type, self.tmpfs_paths), daemon=True)
                    self.cool_down_proc.start()
                elif topic == "stopIteration":
                    if len(self.cache_usage) > 0:
                        np.save('/share/{}_cache_usage.npy'.format(dataset_type), self.cache_usage)
                    self.clear_runtime()
                    
                    terminate = lambda proc: proc.terminate()
                    terminate(self.load_cache_proc)
                    terminate(self.release_cache_proc)
                    terminate(self.mongo_opt_proc)
                    self.mp_manager.shutdown()
                    del self.load_cache_proc, self.release_cache_proc, self.mongo_opt_proc
                    del self.send_idx_queue, self.rcvd_idx_queue, self.mongo_opt_queue, self.cache_usage

                elif topic == "missETags":
                    for etag in pickle.loads(data):
                        self.manager_stub.handle_datamiss(pb.DataMissRequest(cred=self.cred, etag=etag))

                del socks, topic, dataset_type, data
                
    
if __name__ == '__main__':
    client = Client()