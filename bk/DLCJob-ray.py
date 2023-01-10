import os
import json
import bson
import math
import random
import time
import numpy as np
from datetime import datetime
import grpc
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from collections import defaultdict
import ray
import psutil
from multiprocessing import Process, Queue, Lock, Manager
import threading
from typing import Optional, Union, Sequence, Iterable, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t


COOL_DOWN_SEC = 600
class CHUNK_STATUS:
    PREPARE = 0
    ACTIVE = 1
    PENDING = 2
    COOL_DOWN = 3
    INACTIVE = 4

def read_secret(arg):
    path = '/secret/{}'.format(arg)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data

def clear():
    for svr in os.listdir('/runtime/'):
        p = '/runtime/{}'.format(svr)
        if os.path.exists(p) and len(os.listdir(p)) > 0:
            os.system('rm -r {}/*'.format(p))


class DLCJobDataset(Dataset):
    def __init__(self, dtype='train'):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`process`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~DLCJobDataLoader`.
        
        .. note::
        Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self.samples` as Map<Key, Value>.
        Overwriting meth:`process` allows you to replace var:`self.samples` and var:`self.targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            dtype: dataset type, train/validation/test for supervised and unsupervised training/testing
        """
        if dtype not in ["train", "validation", "test"]:
            raise ValueError("invalid dataset type".format(dtype))
        
        self.dtype = dtype
        jobinfo = "/share/{}.json".format(os.environ.get('JOBNAME'))
        while not os.path.exists(jobinfo): pass
        with open(jobinfo, 'rb') as f:
            job_meta = json.load(f)

        self.bucket = job_meta['datasource']['bucket']
        self.qos = job_meta["qos"]
        self.lazy = job_meta['qos']['LazyLoading']
        self.usecache = job_meta['qos']['UseCache']

        self.sample_chunks = []
        self.target_chunks = []
        
        self.samples = {}
        self.targets = {}
        
        if self.usecache:
            from pymongo.mongo_client import MongoClient
            import pandas as pd
            mongo_client = MongoClient(job_meta['mongoUri'])
            self.job_info = mongo_client.Cacher.Job.find_one({"Meta.JobId": job_meta['jobId']})
            self.dataset_col = mongo_client.Cacher.Datasets
            
            # update chunk status
            self.dataset_col.update_many(
                {
                    "Jobs": {"$elemMatch": {"$eq": job_meta['jobId']}},
                    "Category": {"$eq": self.dtype}
                },{
                    "$set": {"Status.code": CHUNK_STATUS.ACTIVE},
                    "$inc": {"Status.active_count": 1}
                }
            )
            
            mfst_etags = self.job_info["ChunkETags"][self.dtype]["manifests"]
            self.manifest = None
            for mfst_etag in mfst_etags:
                mfst_chunk = self.dataset_col.find_one({"ETag": mfst_etag})
                mfst_fpath = "/{}/{}".format(mfst_chunk['Location'], mfst_etag)
                if self.manifest is None:
                    self.manifest = pd.read_csv(mfst_fpath)
                else:
                    self.manifest = pd.concat([self.manifest, pd.read_csv(mfst_fpath)])
            self.loadChunksFromDLCache()
        else:
            import boto3
            cloudSecret = {
                "aws_access_key_id": read_secret('aws_access_key_id'),
                "aws_secret_access_key": read_secret('aws_secret_access_key'),
                "region_name": read_secret('region_name')
            }
            s3_session = boto3.Session(**cloudSecret)
            self.client = s3_session.client('s3')
            self.loadChunksFromCloud()

        self.load_data(partition_index=0)
        
        if self.lazy:
            self.unused_idx = set(list(range(self.__len__())))
            self._miss_queue = Queue()
        Process(target=self.handle_miss, daemon=True).start()

    def try_get_item(self, idx):
        if self.lazy:         
            if self.usecache:
                sample = self.samples[idx]
                try:
                    # t = time.time()
                    X = self.sample_reader(sample)
                    # print('read x use: {}'.format(time.time()-t))
                    Y = None
                    if len(self.targets) > 0:
                        Y = self.targets[idx]
                        if os.path.exists(str(Y)):
                            Y = self.target_reader(Y)
                    read_idx = idx
                except FileNotFoundError:
                    print('miss nfs file {}'.format(sample))
                    while True:
                        if len(self.unused_idx) == 0:
                            sub_idx = random.randint(0, self.__len__() - 1)
                        else:
                            sub_idx = random.choice(list(self.unused_idx))

                        sub_sample, sub_target = self.samples[sub_idx], self.targets[sub_idx]
                        if os.path.exists(sub_sample):
                            etag = sample.split('/')[-1]
                            self._miss_queue.put([[idx, etag], sub_idx])
                            break
                        else:
                            sub_etag = sub_sample.split("/")[-1]
                            self._miss_queue.put([[sub_idx, sub_etag], sub_idx])
                    read_idx = sub_idx
                    X, Y = self.try_get_item(sub_idx)
                
                self.unused_idx.remove(read_idx)
            else:
                X = self.client.get_object(Bucket=self.bucket, Key=self.samples[idx])['Body'].read()
                Y = None
                if len(self.targets) > 0:
                    Y = self.targets[idx]
                    if 'Contents' in self.client.list_objects(Bucket=self.bucket, Prefix=str(Y)):
                        Y = self.client.get_object(Bucket=self.bucket, Key=Y)['Body'].read()            
            return (X, Y)
        else:
            return (self.samples[idx], self.targets[idx])

    def handle_miss(self):
        manager_uri = "dlcpod-manager:50051"
        self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
        channel = grpc.insecure_channel(manager_uri)
        stub = pb_grpc.DataMissStub(channel)
        def swap(x, y):
            x, y = y, x
            
        try:
            while True:
                info = self._miss_queue.get(block=True)
                miss_idx, miss_etag = info[0]
                sub_idx = info[1]
                swap(self.samples[miss_idx], self.samples[sub_idx])
                swap(self.targets[miss_idx], self.targets[sub_idx])
                stub.call(pb.DataMissRequest(cred=self.cred, etag=miss_etag))
        except KeyboardInterrupt:
            self._miss_queue.cancel_join_thread()
            self._miss_queue.close()
            channel.close()
            
    def loadChunksFromDLCache(self):
        """Load all chunks (MongoDB query result) of the given dataset. 
        
        In the LazyLoading mode, there is only 1 group because no memory pressure.
        Otherwise, we group chunks to make them fit the MaxPartMill constraint.

        Returns:
            None: initialize the self.sample_chunks and self.target_chunks
        """
        chunk_etags = self.job_info['ChunkETags'][self.dtype]
        
        def helper(etags):
            chunks = []
            if etags:
                chunks_iter = self.dataset_col.find({"ETag": {"$in": etags}})
                chunks = [chunk for chunk in chunks_iter]
            return chunks
    
        self.sample_chunks = helper(chunk_etags['samples'])
        if self.job_info['QoS']['LazyLoading']:
            self.sample_chunks = [self.sample_chunks]
        self.target_chunks = helper(chunk_etags['targets'])
        
    # dataset shouldn't be compressed when using this function
    def loadChunksFromCloud(self):
        paginator = self.client.get_paginator('list_objects_v2')       
        def load_pages(keys):
            if keys:
                pages = []
                for k in keys:
                    pages.extend(paginator.paginate(Bucket=self.bucket, Prefix=k))
                return pages
            return None

        keys = self.job_info['Datasource']['keys'][self.dtype]
        
        def helper(keys):
            chunks = []
            if keys:
                pages = load_pages(keys)
                for i in range(len(pages)):
                    tmp = []
                    for j in range(len(pages[i]['Contents'])):
                        dataobj = pages[i]["Contents"][j]
                        tmp.append(dataobj)
                    chunks.append(tmp)
            return chunks
        
        self.sample_chunks = helper(keys['samples'])
        self.target_chunks = helper(keys['targets'])  

    def load_data(self, partition_index):
        """Load file paths or actual data in the given partition
        
        Initialize the self.samples and self.targets, where self.samples is a dict with format {key: file_path/data}
        user performs data (X and y) processing in the process function, that convert self.samples ans self.targets
        from dict to iteratable X, y
        
        If a chunk is a compressed, we generate a dummy key for individual files.
        We start operating on individual data items from this function. Before this, all operations are on chunks.
        
        Samples are matched with corresponding targets based on the manifest file provided by user, which specifies the 
        mappings between X and y.
        
        Only file-based datasets might have the manifest file.
        
        Args:
            partition_index (int): file-based (LazyLoading) dataset only has one partition, so the partition_index = 0
                             tabular dataset might split file to multiple partitions
        """
        
        def helper(chunks, reader):
            data = {}
            for chunk in chunks[partition_index]:
                if not chunk: 
                    continue
                key, loc = chunk['Key'], chunk['Location']
                if self.usecache:
                    chunk_path = '/{}/{}'.format(loc, chunk['ChunkETag'])
                    # decompressed folder, so key has the .tar.gz extension
                    if os.path.isdir(chunk_path):
                        # We assume data won't be immediately deleted after being downloaded by Manager.
                        for root, dirs, files in os.walk(chunk_path):
                            for name in files:
                                p = os.path.join(root, name)
                                dummy_key = key + p.replace(chunk_path, '')
                                data[dummy_key] = p if self.lazy else reader(p)
                    else:
                        data[key] = chunk_path if self.lazy else reader(chunk_path)
                else:
                    data[key] = key
            return data
        
        '''self.samples and self.targets are dictionaries with format {'cloud_key': 'nfs_path'}
        '''
        self.samples = helper(self.sample_chunks, self.sample_reader)
        if self.target_chunks:
            self.targets = helper(self.target_chunks, self.target_reader)
        
        self.samples, self.targets = self.process()

    def sample_reader(self, path: str = None, raw_bytes: bytes = None):
        """this function defines the logic of reading sample data (X) from a file

        Args:
            path (string): read from a file
            raw_bytes (bytes): read from raw bytes
            
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def target_reader(self, path: str = None, raw_bytes: bytes = None):
        """this function defines the logic of reading target data (Y) from a file

        Args:
            path (string): read from a file
            raw_bytes (bytes): read from raw bytes
        """
        return
    
    def process(self) -> Tuple[List, List]:
        """process self.samples ans self.target

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        """get the sample and target at the given index
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    

@ray.remote
class DLCJobDataLoader(object):
    def __init__(self, dataset: DLCJobDataset, 
                 batch_size: Optional[int] = 1, shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        """
        Data loader. Combines a dataset and a sampler, and provides an iterable over
        the given dataset.

        The :class:`~torch.utils.data.DataLoader` supports both map-style and
        iterable-style datasets with single- or multi-process loading, customizing
        loading order and optional automatic batching (collation) and memory pinning.

        See :py:mod:`torch.utils.data` documentation page for more details.

        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
            sampler (Sampler or Iterable, optional): defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
            batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
                returns a batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
                and :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn (callable, optional): If not ``None``, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading. (default: ``None``)
            generator (torch.Generator, optional): If not ``None``, this RNG will be used
                by RandomSampler to generate random indexes and multiprocessing to generate
                `base_seed` for workers. (default: ``None``)
            prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
                in advance by each worker. ``2`` means there will be a total of
                2 * num_workers samples prefetched across all workers. (default: ``2``)
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. (default: ``False``)


        .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                    cannot be an unpicklable object, e.g., a lambda function. See
                    :ref:`multiprocessing-best-practices` on more details related
                    to multiprocessing in PyTorch.

        .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                    When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                    it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                    rounding depending on :attr:`drop_last`, regardless of multi-process loading
                    configurations. This represents the best guess PyTorch can make because PyTorch
                    trusts user :attr:`dataset` code in correctly handling multi-process
                    loading to avoid duplicate data.

                    However, if sharding results in from threading import Thread

        .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                    :ref:`data-loading-randomness` notes for random seed related questions.
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.num_batches = math.ceil(len(self.dataset.samples)/batch_size)
        
        self.lazy = self.dataset.qos['LazyLoading']
        self.partition_index = 0
        clear()
        
        self._rcvd_idx = -1
        self.req_time = []
        self.load_time = []
        self.lock = Lock()
        
        self._active_workers = 0
        t = time.time()
        self._init_loader(first_epoch=True)
        print('init loader time: {}'.format(time.time() - t))
        
        self.data_queue = Queue()

        self.cache_controller_proc = Process(target=self.cache_controller)
        self.cool_down_proc = None
        
    def _init_loader(self, first_epoch=True):
        if first_epoch or self.shuffle:
            # set the num_workers=0 to use the main process to load data
            torch_loader = DataLoader(self.dataset, self.batch_size, self.shuffle, self.sampler, self.batch_sampler, 0, self.collate_fn, 
                                           self.pin_memory, self.drop_last, self.timeout, self.worker_init_fn, self.multiprocessing_context, self.generator, 
                                           prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
            self._index_sampler = [idx for idx in iter(torch_loader._index_sampler)]
            self._idx_iter = iter(list(range(self.num_batches)))
            
            # update chunk status to ACTIVE
            etags = []
            tmp = self.dataset.sample_chunks.copy()
            tmp.extend(self.dataset.target_chunks)
            for part in tmp:
                for chunk in part:
                    etags.append(chunk['ChunkETag'])
            now = datetime.utcnow().timestamp()
            self.dataset.dataset_col.update_many(
                {"ChunkETag": {"$in": etags}}, 
                {
                    "$set": { "Status.code": CHUNK_STATUS.ACTIVE},
                    "$inc": {"Status.active_count": 1},
                    "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}
                }
            )
    
    def _worker_loop(self, batch_idx):
        samples = []
        targets = []
        for item_idx in self._index_sampler[batch_idx]:
            print(item_idx)
            item = self.dataset.try_get_item(item_idx)
            samples.append(item[0])
            targets.append(item[1])
        data = (samples, targets)
        return data

    def cache_controller(self):
        send_idx = 0
        while send_idx < self.num_batches:
            if send_idx - self._rcvd_idx >= self.num_workers:
                continue
            futures = []
            for _ in range(self.num_workers):
                futures.append(self._worker_loop.remote(send_idx))
                send_idx += 1
            print(futures)
            for data in ray.get(futures):
                self.data_queue.put(ray.put(data))

    def expire_cache(self):
        time.sleep(COOL_DOWN_SEC)
        etags = []
        for sample_path in self.dataset.samples:
            etags.append(sample_path.split('/')[-1])
        for target_path in self.dataset.targets:
            if target_path:
                etags.append(target_path.split("/")[-1])
        self.dataset.dataset_col.update_many(
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
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.num_batches
    
    def _load_data_from_cache(self):
        batch_idx = next(self._idx_iter)
        obj_ref = self.data_queue.get()
        data = ray.get(obj_ref)
        return data
    
    def __next__(self):
        try:
            if self._rcvd_idx == -1:
                self._rcvd_idx = 0
                self.cache_controller_proc.start()
            elif self._rcvd_idx == 0:
                self._init_loader(first_epoch=False)
            elif self._rcvd_idx == self.num_batches:
                raise StopIteration
            
            t = time.time()
            data = self._load_data_from_cache()
            self.req_time.append(time.time()-t)
            self._rcvd_idx += 1
        except StopIteration:
            print('raise StopIteration Exception....')
            self._rcvd_idx = 0
            
            # epoch is down
            if self.partition_index == len(self.dataset.sample_chunks)-1:
                self.partition_index = 0
                
                # update chunk status to COOL_DOWN
                etags = []
                all_chunks = self.dataset.sample_chunks.copy()
                all_chunks.extend(self.dataset.target_chunks)
                for part in all_chunks:
                    for chunk in part:
                        etags.append(chunk['ChunkETag'])

                now = datetime.utcnow().timestamp()
                self.dataset.dataset_col.update_many(
                    {
                        "ChunkETag": {"$in": etags},
                        "Status.active_count": 1
                    },
                    {"$set": {
                        "Status.code": CHUNK_STATUS.INACTIVE,
                        "Status.active_count": 0,
                        "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
                    }
                )
                self.dataset.dataset_col.update_many(
                    {
                        "ChunkETag": {"$in": etags},
                        "Status.active_count": {"$gt": 1}
                    },
                    {
                        "$inc": {"Status.active_count": -1},
                        "$set": {
                            "Status.code": CHUNK_STATUS.INACTIVE,
                            "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)
                        }
                    }
                )
                
                if self.cool_down_proc is not None and self.cool_down_proc.is_alive():
                    self.cool_down_proc.terminate()
                self.cool_down_proc = Process(target=self.expire_cache, daemon=True)
                self.cool_down_proc.start()
                raise StopIteration
            else:                    
                # data in the current part have been consumed    
                self.partition_index += 1
                clear()
                self.dataset.load_data(self.partition_index)
                self._init_loader()
                data = self._load_data_from_cache()
                
        return data