import os
import json
import bson
import math
import statistics
import random
import time
from datetime import datetime
import grpc
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
import threading
import queue
from typing import Optional, Union, Sequence, Iterable, List, Tuple
import torch
from torch import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, RandomSampler, SequentialSampler
import torch.utils.data._utils.pin_memory as pm
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t, default_collate


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
            self.db = mongo_client.Cacher   
            self.job_info = self.db.Job.find_one({"Meta.JobId": job_meta['jobId']})
            manifest_etags = self.job_info["ChunkETags"][self.dtype]["manifests"]
            self.manifest = None
            for manifest_etag in manifest_etags:
                manifest_chunk = self.db.Datasets.find_one({"ETag": manifest_etag})
                manifest_fpath = "/{}/{}".format(manifest_chunk['Location'], manifest_etag)
                if self.manifest is None:
                    self.manifest = pd.read_csv(manifest_fpath)
                else:
                    self.manifest = pd.concat([self.manifest, pd.read_csv(manifest_fpath)])
            self.load_chunks()
        else:
            import boto3
            cloudSecret = {
                "aws_access_key_id": read_secret('aws_access_key_id'),
                "aws_secret_access_key": read_secret('aws_secret_access_key'),
                "region_name": read_secret('region_name')
            }
            s3_session = boto3.Session(**cloudSecret)
            self.client = s3_session.client('s3')
            self.load_chunks_from_cloud()

        self.load_data(partition_index=0)
        self.handle_miss_thread = None

    def __default_get_item__(self, index):            
        if self.lazy:         
            if self.usecache:
                if self.handle_miss_thread is None:
                    self._miss_queue = queue.Queue()
                    self.handle_miss_thread = threading.Thread(target=self.handle_miss, daemon=True)
                    self.handle_miss_thread.start()
                
                sample = self.samples[index]
                target = self.targets[index]
                try:
                    X = self.sample_reader(sample)
                    Y = self.target_reader(target)
                except FileNotFoundError:
                    print('miss nfs file {}'.format(sample))
                    etag = sample.split('/')[-1]
                    self._miss_queue.put(etag)
                    sub_idx = random.randint(0, len(self) - 1)
                    X, Y = self.__default_get_item__(sub_idx)
            else:
                X = self.client.get_object(Bucket=self.bucket, Key=self.samples[index])['Body'].read()
                Y = None
                if len(self.targets) > 0:
                    Y = self.targets[index]
                    if 'Contents' in self.client.list_objects(Bucket=self.bucket, Prefix=str(Y)):
                        Y = self.client.get_object(Bucket=self.bucket, Key=Y)['Body'].read()            
            return (X, Y) if Y is not None else X
        else:
            return (self.samples[index], self.targets[index])

    def handle_miss(self):
        manager_uri = "dlcpod-manager:50051"
        self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
        channel = grpc.insecure_channel(manager_uri)
        stub = pb_grpc.DataMissStub(channel)

        try:
            while True:
                miss_etag = self._miss_queue.get(block=True)
                stub.call(pb.DataMissRequest(cred=self.cred, etag=miss_etag))
        except KeyboardInterrupt:
            channel.close()
            
    def load_chunks(self):
        """Load all chunks (MongoDB query result) of the given dataset. 
        
        In the LazyLoading mode, there is only 1 group because no memory pressure.
        Otherwise, we group chunks to make them fit the MaxPartMill constraint.

        Returns:
            None: initialize the self.sample_chunks and self.target_chunks
        """
        
        self.chunk_etags = []
        def helper(etags):
            chunks = []
            if etags:
                chunks_iter = self.db.Datasets.find({"ETag": {"$in": etags}})
                for chunk in chunks_iter:
                    chunks.append(chunk)
                    self.chunk_etags.append(chunk['ChunkETag'])
            return chunks
    
        chunk_etags = self.job_info['ChunkETags'][self.dtype]
        self.sample_chunks = helper(chunk_etags['samples'])
        if self.job_info['QoS']['LazyLoading']:
            self.sample_chunks = [self.sample_chunks]
        self.target_chunks = helper(chunk_etags['targets'])
        
    # dataset shouldn't be compressed when using this function
    def load_chunks_from_cloud(self):
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
        
    def sample_reader(self, sample_item):
        raise NotImplementedError
    
    def target_reader(self, target_item):
        raise NotImplementedError
    
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
        return self.__default_get_item__(index)
    
    def __len__(self) -> int:
        return len(self.samples)


class WorkerInfo:
    def __init__(self, worker_id=None, status=True):
        self.worker_id = worker_id
        self.status = status
        self.next = None
    

class WorkerQueueIndexCycle:
    def __init__(self) -> None:
        super().__init__()
        self.__head = None
        
    def is_empty(self):
        return not self.__head
        
    def __next__(self):        
        while not self.__ptr.status:
            self.__ptr = self.__ptr.next
        worker_queue_idx = self.__ptr.worker_id
        self.__ptr = self.__ptr.next
        return worker_queue_idx

    def __iter__(self):
        return self
    
    # add from the head
    def add(self, worker_id, status=True):
        w = WorkerInfo(worker_id, status)
        if self.is_empty():
            self.__head = w
            w.next = self.__head
            self.__ptr = self.__head
            return
        cur = self.__head
        while cur.next != self.__head:
            cur = cur.next
        cur.next = w
        w.next = self.__head
        self.__head = w
    
    # append from the end
    def append(self, worker_id, status=True):
        if self.is_empty():
            self.add(worker_id, status)
            return
        cur = self.__head
        while cur.next != self.__head:
            cur = cur.next
        w = WorkerInfo(worker_id, status)
        cur.next = w
        w.next = self.__head
    
    def num_active_workers(self):
        if self.is_empty():
            return 0
        n = self.__head.status
        cur = self.__head
        while cur.next != self.__head:
            n += cur.status
            cur = cur.next
        return n
    
    def set_status(self, index, value):
        if index < 0:
            raise IndexError
        if index > self.length() - 1:
            raise IndexError
        cur = self.__head
        for _ in range(index):
            cur = cur.next
        cur.status = value
    
    def remove(self, index):
        if index < 0:
            raise IndexError
        if index > self.length() - 1:
            raise IndexError
        cur = self.__head
        prev = None
        for i in range(index):
            prev = cur
            cur = cur.next
        if cur == self.__head:
            prev = self.__head
            while prev.next != self.__head:
                prev = prev.next
            if prev == self.__head:
                self.__head = None
                return
            prev.next = self.__head.next
            self.__head = self.__head.next
            return
        prev.next = cur.next
    
    
    
class DLCJobDataLoader(object):
    def __init__(self, dataset: DLCJobDataset, 
                 batch_size: Optional[int] = 1, shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False, tune_num_workers_frequency: int = 10, max_num_workers: int = None):
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
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        
        if collate_fn is None:
            self.collate_fn = default_collate
            
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.tune_num_workers_frequency = tune_num_workers_frequency
        if max_num_workers is None:
            self.max_num_workers = mp.cpu_count() // 2
        else:
            self.max_num_workers = max_num_workers
            
        if self.sampler is None:
            if self.shuffle:
                self.sampler = RandomSampler(self.dataset)
            else:
                self.sampler = SequentialSampler(self.dataset)
        
        if self.batch_size > 0 and self.batch_sampler is None:
            self._index_sampler = BatchSampler(self.sampler, self.batch_size, self.drop_last)

        self.num_batches = len(self._index_sampler)
        self.lazy = self.dataset.qos['LazyLoading']
        self._partition_idx = 0
        
        self._index_queues = []
        self._worker_queue_idx_cycle = WorkerQueueIndexCycle()
        self._worker_result_queue = mp.Queue()    
        self._workers = []
        self._workers_down_events = []
        self._workers_status = []
        self._active_workers = 0
        
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
        for _ in range(self.num_workers):
            self._spawn_worker()
        
        self.pin_memory = self.pin_memory and torch.cuda.is_available()
        if self.pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            pin_memory_thread = threading.Thread(
                target=pm._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue
        
        self._cool_down_proc = None
        self._mongo_operation_queue = queue.Queue()
        threading.Thread(target=self.async_mongo_operator).start()
        self._reset()

    def async_mongo_operator(self):
        try:
            while True:
                collection, func, opertion = self._mongo_operation_queue.get(block=True)
                if func == 'update_many':
                    self.dataset.db[collection].update_many(**opertion)
        except KeyboardInterrupt:
            pass

    def _spawn_worker(self):
        for i in range(len(self._workers_status)):
            if not self._workers_status[i]:
                self._mark_worker_as_available(i)
                return

        idx_queue = mp.Queue()
        idx_queue.cancel_join_thread()
        done_event = mp.Event()
        worker_id = len(self._workers_status)
        w = mp.Process(target=self._worker_loop, 
                       args=(self.dataset, worker_id, idx_queue, self._worker_result_queue, done_event, self._base_seed))
        w.daemon = True
        w.start()
        self._index_queues.append(idx_queue)
        self._workers.append(w)
        
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object.
        self._workers_status.append(True)
        self._worker_queue_idx_cycle.append(self._active_workers)
        self._workers_down_events.append(done_event)
        self._active_workers += 1
        print('add worker {}'.format(self._active_workers))
    
    def _pause_worker(self):
        for i in range(len(self._workers_status)):
            if self._workers_status[i]:
                self._workers_status[i] = False
                self._worker_queue_idx_cycle.set_status(i, False)
                # print('pause worker {}'.format(i))
                self._active_workers -= 1
                break
    
    def _mark_worker_as_available(self, worker_id):
        assert not self._workers_status[worker_id]
        self._workers_status[worker_id] = True
        self._worker_queue_idx_cycle.set_status(worker_id, True)
        self._active_workers += 1
        # print('reactive worker {}'.format(worker_id))
        
    def _reset(self):    
        self._sampler_iter = iter(self._index_sampler)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
            
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        
        self._last_iter_time = None
        
        self._req_time = []
        self._load_time = []
        
        # prime the prefetch loop
        for _ in range(self.prefetch_factor * self.num_workers):
            self._try_put_index()

    def _worker_loop(self, dataset, worker_id, index_queue, data_queue, done_event, base_seed):
        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        
        try:
            while True:
                batch_idx, batch, idx_req_time = index_queue.get(block=True)
                # print('worker {} get batch {} from queue'.format(worker_id, batch_idx))
                data = []
                for idx in batch:
                    data.append(dataset[idx])
                data = self.collate_fn(data)
                data_queue.put((batch_idx, data, time.time()-idx_req_time))
                # print('worker {} put batch {} into queue'.format(worker_id, batch_idx))
                del batch_idx, batch, data  # save memory
        except KeyboardInterrupt:
            pass

        if done_event.is_set():
            data_queue.cancel_join_thread()
            data_queue.close()

    def _try_put_index(self):
        # calculate the # of index should be put
        if self._rcvd_idx > 0 and self._rcvd_idx % self.tune_num_workers_frequency == 0:
            print('active workers: {}'.format(self._active_workers))
            req_interval = statistics.median(self._req_time)
            load_interval = statistics.median(self._load_time)
            # print('req: {}, load: {}'.format(req_interval, load_interval))
            k = math.ceil(load_interval/req_interval) - self._active_workers
            # print('add {} workers'.format(k))
            if k > 0:
                for _ in range(k):
                    if self._active_workers > self.max_num_workers:
                        break
                    self._spawn_worker()
            elif k < 0:
                for _ in range(-k):
                    self._pause_worker()
            self._req_time = []
            self._load_time = []
        
        try:
            batched_idxs = next(self._sampler_iter)
        except StopIteration:
            return

        worker_queue_idx = next(self._worker_queue_idx_cycle)
        self._index_queues[worker_queue_idx].put((self._send_idx, batched_idxs, time.time()))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self._index_sampler)
    
    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        return data
    
    def _next_data(self):
        # while self._rcvd_idx < self._send_idx:
        #     info = self._task_info[self._rcvd_idx]
        #     worker_id = info[0]
        #     if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
        #         break
        #     del self._task_info[self._rcvd_idx]
        #     print('del task info {}, len {}'.format(self._rcvd_idx, len(info)))
        #     self._rcvd_idx += 1
        # else:
        #     print(self._rcvd_idx, self._send_idx)
        #     raise StopIteration

        if self._rcvd_idx == 0:
            # update chunk status to ACTIVE
            now = datetime.utcnow().timestamp()
            self._mongo_operation_queue.put(('Datasets', 'update_many', 
                                            {
                                                "filter": {"ChunkETag": {"$in": self.dataset.chunk_etags}}, 
                                                "update": {
                                                    "$set": { "Status.code": CHUNK_STATUS.ACTIVE},
                                                    "$inc": {"Status.active_count": 1},
                                                    "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}}
                                            }))
            if self._cool_down_proc is not None and self._cool_down_proc.is_alive():
                self._cool_down_proc.terminate()
        elif self._rcvd_idx >= self._send_idx:
            raise StopIteration
        
        # Now `self._rcvd_idx` is the batch index we want to fetch

        while True:
            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert self._tasks_outstanding > 0
            
            try:
                idx, data, dur = self._data_queue.get(timeout=5.0)
            except queue.Empty:
                print('need batch {}, empty data queue, task info: {}: {}'.format(self._rcvd_idx, list(self._task_info.keys()), [len(self._task_info[x]) for x in self._task_info]))
                continue
            
            self._load_time.append(dur)
            self._tasks_outstanding -= 1

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)
            
    def __next__(self):
        with torch.autograd.profiler.record_function(self._profile_name):
            try:
                if self._sampler_iter is None:
                    print('self._sampler_iter is None')
                    raise StopIteration
                
                if self._last_iter_time is not None:
                    self._req_time.append(time.time()-self._last_iter_time)

                data = self._next_data()
                self._last_iter_time = time.time()
            except StopIteration:
                print('raise StopIteration Exception....')
                
                # epoch is down
                if self._partition_idx == len(self.dataset.sample_chunks)-1:
                    self._partition_idx = 0
                    
                    # update chunk status to COOL_DOWN
                    now = datetime.utcnow().timestamp()
                    self._mongo_operation_queue.put(("Datasets", "update_many", 
                                                    {
                                                        "filter": {
                                                            "ChunkETag": {"$in": self.dataset.chunk_etags},
                                                            "Status.active_count": 1
                                                        },
                                                        "update": {"$set": {
                                                            "Status.code": CHUNK_STATUS.INACTIVE,
                                                            "Status.active_count": 0,
                                                            "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)}
                                                        }
                                                    }))
                    self._mongo_operation_queue.put(("Datasets", "update_many", 
                                                    {
                                                        "filter": {
                                                            "ChunkETag": {"$in": self.dataset.chunk_etags},
                                                            "Status.active_count": {"$gt": 1}
                                                        },
                                                        "update": {
                                                            "$inc": {"Status.active_count": -1},
                                                            "$set": {
                                                                "Status.code": CHUNK_STATUS.INACTIVE,
                                                                "Status.cool_down_init": bson.timestamp.Timestamp(int(now), inc=1)
                                                            }
                                                        }
                                                    }))
                    
                    self._cool_down_proc = mp.Process(target=self.expire_chunks, daemon=True)
                    self._cool_down_proc.start()
                    self._reset()
                    raise StopIteration
                else:                    
                    # data in the current part have been consumed    
                    self._partition_idx += 1
                    self.dataset.load_data(self._partition_idx)
                    self._reset()
                    data = self._next_data()
                    self._last_iter_time = time.time()
            return data
    
    def expire_chunks(self):
        time.sleep(COOL_DOWN_SEC)   
        self._mongo_operation_queue.put(("Datasets", "update_many", 
                                         {
                                             "filter": {
                                                 "ChunkETag": {"$in": self.dataset.chunk_etags},
                                                 "Status.code": CHUNK_STATUS.COOL_DOWN
                                             },
                                             "update": {
                                                 "$set":{
                                                     "Status.code": CHUNK_STATUS.INACTIVE,
                                                     "Status.active_count": 0
                                                 }
                                             }
                                         }))
        
    # def _shut_down_worker(self, n = 1):
    #     count = 0
    #     for i in range(len(self._workers_status)):
    #         if count == n:
    #             break
    #         if self._workers_status[i]:
    #             self._workers_down_events[i].set()
    #             self._mark_worker_as_unavailable(i, True)
    #             count += 1
                
    # def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
    #         # Mark a worker as having finished its work e.g., due to
    #     # exhausting an `IterableDataset`. This should be used only when this
    #     # `_MultiProcessingDataLoaderIter` is going to continue running.

    #     assert self._workers_status[worker_id] or (self.persistent_workers and shutdown)

    #     # Signal termination to that specific worker.
    #     q = self._index_queues[worker_id]
    #     # Indicate that no more data will be put on this queue by the current
    #     # process.
    #     q.put(None)

    #     # Note that we don't actually join the worker here, nor do we remove the
    #     # worker's pid from C side struct because (1) joining may be slow, and
    #     # (2) since we don't join, the worker may still raise error, and we
    #     # prefer capturing those, rather than ignoring them, even though they
    #     # are raised after the worker has finished its job.
    #     # Joinning is deferred to `_shutdown_workers`, which it is called when
    #     # all workers finish their jobs (e.g., `IterableDataset` replicas) or
    #     # when this iterator is garbage collected.

    #     self._workers_status[worker_id] = False

    #     assert self._workers_down_events[worker_id].is_set() == shutdown

    # def _shutdown_workers(self):
    #     if not self._shutdown:
    #         self._shutdown = True
    #         while len(self._workers) > 0:
    #             self._shut_down_worker()

    # def __del__(self):
    #     self._shutdown_workers()
