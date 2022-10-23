import os
import json
import concurrent
import math
import random
import numpy as np
import zmq
import multiprocessing
from multiprocessing import Process, Queue, Condition, Lock
from typing import Optional, Union, Sequence, Iterable, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t


jobinfo = "/share/{}.json".format(os.environ.get('JOBNAME'))
init_channel = 'ipc:///share/init.ipc'
ipc_channel = 'ipc:///share/runtime.ipc'


class DLCJobDataset(Dataset):
    def __init__(self, keys: List[str] = None):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__convert__`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~DLCJobDataLoader`.
        
        .. note::
        Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self.data` as Map<Key, Value>.
        Overwriting meth:`__convert__` allows you to replace var:`self.data` and var:`self.targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            keys (List, optional): a list of bucket keys. Defaults to None, meaning loading all keys in the bucket.
            shuffle (Bool, optional): default False
        """
        self.keys = keys
        while not os.path.exists(jobinfo): pass
        with open(jobinfo, 'rb') as f:
            job_meta = json.load(f)
        self.qos = job_meta['qos']
        self.bucket = job_meta['dataSource']['bucket']
        self.chunks = []
        
        if self.qos['UseCache']:
            from pymongo.mongo_client import MongoClient
            mongo_client = MongoClient(job_meta['mongoUri'])
            self.jobId = job_meta['jobId']
            self.job_col = mongo_client.Cacher.Job
            self.dataset_col = mongo_client.Cacher.Datasets
            self.loadKeysFromDLCache()
        else:
            import configparser, boto3
            parser = configparser.ConfigParser()
            parser.read('/secret/client.conf')
            s3auth = parser['AWS']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            self.client = s3_session.client('s3')
            self.loadKeysFromCloud()
        
        self.nfsFilePaths = []
        self.targets = None
        self.load_data(chunk_idx=0)
        
        self.miss_queue = Queue()
        self.lock = Lock()
        self.unused_idx = set(list(range(self.__len__())))
                    
    def get_data(self, idx):
        if self.qos['LazyLoading']:
            if len(self.unused_idx) == 0:
                self.unused_idx = set(list(range(self.__len__())))
            return self.read(self.data[idx], idx)
        else:
            return self.data[idx]
    
    def get_target(self, index):
        return self.targets[index]

    def loadKeysFromDLCache(self):
        maxmem = self.qos['MaxMemoryMill']*1e6  
        etags = self.job_col.find_one({"Meta.JobId": self.jobId})['ETags']
        chunks = [chunk for chunk in self.dataset_col.find({"ChunkETag": {"$in": etags}})]
                    
        if maxmem == 0 or self.qos['LazyLoading']:
            self.chunks.append(chunks)
        else:
            total_size = 0
            self.chunks.append([])
            for chunk in chunks:
                s = int(chunk['Size'])
                total_size += s
                # any single chunk should be smaller than the MaxMemory
                if total_size >= maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('Object {} size is greater than assigned MaxMemory.'.format(chunk['Key']))
                else:
                    self.chunks[-1].append(chunk)
        
    def loadKeysFromCloud(self):
        paginator = self.client.get_paginator('list_objects_v2')
        if self.keys is not None:
            pages = []
            for k in self.keys:
                pages.extend(paginator.paginate(Bucket=self.bucket, Prefix=k))
        else:
            pages = paginator.paginate(Bucket=self.bucket)
        
        maxmem = self.qos['MaxMemoryMill']*1e6
        maxmem = math.inf if maxmem==0 else maxmem
        total_size = 0
        self.chunks.append([])
        for page in pages:
            for item in page['Contents']:
                s = int(item['Size'])
                total_size += s
                if total_size > maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(item['Key']))
                else:
                    self.chunks[-1].append(item)
    
    def read(self, chunk, idx=None):
        key = chunk['Key']
        if self.qos['UseCache']:
            etag, loc = chunk['ChunkETag'], chunk['Location']
            nfs_path = '/{}/{}'.format(loc, etag)
            tmpfs_path = '/runtime{}'.format(nfs_path)
            
            read_idx = idx
            read_val = None
            
            # 3-level data hit
            if os.path.exists(tmpfs_path):
                with open(tmpfs_path, 'rb') as f:
                    read_val = f.read()
                # print("read tmpfs file {}".format(tmpfs_path))
            elif os.path.exists(nfs_path):
                with open(nfs_path, 'rb') as f:
                    read_val = f.read()
                # print("miss tmpfs file {}".format(tmpfs_path))
            else:
                # substitutable cache hit
                # 1. randomly select an unused data point
                # 2. replace the idx with sub_idx
                # 3. socket in data loader notify client to update data access sequence
                print('miss nfs file {}'.format(nfs_path))
                while True:
                    # TODO： 当有多个worker同时load数据时，sub_idx可能重复， 但是如果上锁，会让并行worker串行，影响速度
                    if len(self.unused_idx) == 0:
                        sub_idx = random.randint(0, self.__len__()-1)
                    else:
                        sub_idx = random.choice(list(self.unused_idx))
                    sub_etag, sub_loc = self.data[sub_idx]['ChunkETag'], self.data[sub_idx]['Location']
                    p =  '/{}/{}'.format(sub_loc, sub_etag)
                    if os.path.exists(p):
                        self.miss_queue.put([[idx, etag], [sub_idx, sub_etag]])
                        read_idx = sub_idx
                        break
                    else:
                        self.miss_queue.put([[sub_idx, sub_etag], [sub_idx, sub_etag]])
                read_val = self.read(self.data[sub_idx], sub_idx)
            
            if read_idx is not None and read_idx in self.unused_idx:
                self.unused_idx.remove(read_idx)
            return read_val
        else:
            return self.client.get_object(Bucket=self.bucket, Key=key)['Body'].read()

    def load_data(self, chunk_idx):
        """Load file paths or actual data in the given chunk 
        initialize the self.data and self.keys, where self.data is a dict with format {key: dataobj}
        user performs data (X and y) pre-processing in the __convert__ function, that convert self.data from
        dict to iteratable X, y
        
        Args:
            chunk_idx (int): file-based (LazyLoading) dataset only has one chunk, so the chunk_idx = 0
                             tabular dataset split file to multiple chunks
        """
        self.data = {}
        self.keys = []
        if self.qos['LazyLoading']:
            # file-based dataset only has one chunk, so the chunk_idx = 0
            for dataobj in self.chunks[chunk_idx]:
                self.keys.append(dataobj['Key'])
                self.nfsFilePaths.append('/{}/{}'.format(dataobj['Location'], dataobj['ChunkETag']))
                self.data[dataobj['Key']] = dataobj
        else:
            # Normal mode fits tabular or block datasets, so we load a subset of the original dataset here
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for dataobj in self.chunks[chunk_idx]:
                    futures.append(executor.submit(self.read, dataobj))
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    dataobj = self.chunks[chunk_idx][i]
                    self.keys.append(dataobj['Key'])
                    self.nfsFilePaths.append('/{}/{}'.format(dataobj['Location'], dataobj['ChunkETag']))
                    self.data[dataobj['Key']] = future.result()
        
        self.data, self.targets = self.__convert__()
    
    def __convert__(self) -> Tuple[List, List]:
        """convert self.data ans self.target

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    
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
        self.num_batches = math.ceil(len(self.dataset.data)/batch_size)
        self.lazy = self.dataset.qos['LazyLoading']
        self.chunk_idx = 0
        self.clear()
        
        context = zmq.Context()
        self.socket_req = context.socket(zmq.REQ)
        self.socket_req.connect(init_channel)
        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.connect(ipc_channel)
        
        self._init_loader(first_iter=True)
        Process(target=self.handle_miss, daemon=True).start()
    
    def _init_loader(self, first_iter=False):      
        if first_iter or self.shuffle:
            self.torch_loader = DataLoader(self.dataset, self.batch_size, self.shuffle, self.sampler, self.batch_sampler, self.num_workers, self.collate_fn, 
                                  self.pin_memory, self.drop_last, self.timeout, self.worker_init_fn, self.multiprocessing_context, self.generator, 
                                  prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
            file_paths = np.array(self.dataset.nfsFilePaths)
            self.batchedNfsPaths = [file_paths[idx].tolist() for idx in iter(self.torch_loader._index_sampler)]
            # client will copy the first batch when receive init msg
            self.socket_req.send_multipart([b'init', json.dumps({
                "paths": self.batchedNfsPaths,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor}).encode('utf-8')])
            self.socket_req.recv()
        self.loader = iter(self.torch_loader)
    
    @staticmethod
    def clear():
        for svr in os.listdir('/runtime/'):
            p = '/runtime/{}'.format(svr)
            if os.path.exists(p) and len(os.listdir(p)) > 0:
                os.system('rm -r {}/*'.format(p))
    
    def handle_miss(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.connect(ipc_channel)
        while True:
            info = self.torch_loader.dataset.miss_queue.get()
            if info is not None:
                miss_idx, miss_etag = info[0]
                sub_idx, sub_etag = info[1]
                msg = "{}:{} {}:{}".format(miss_idx, miss_etag, sub_idx, sub_etag)
                socket.send_multipart([b'dataMiss', msg.encode('utf-8')])
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.loader.__len__()
    
    def __next__(self):
        try:
            # prefetch the next batch
            pre_idx = list(range(self.loader._send_idx+(self.loader._rcvd_idx != 0), min(self.loader._send_idx+self.prefetch_factor, len(self.batchedNfsPaths))))    
            pre_idx = [str(idx) for idx in pre_idx]
            if len(pre_idx) > 0:
                self.socket_pub.send_multipart([b'prefetch', ','.join(pre_idx).encode('utf-8')])
            
            # release the last batch
            if self.loader._rcvd_idx > 0:
                self.socket_pub.send_multipart([b'releaseCache', str(self.loader._rcvd_idx-1).encode('utf-8')])
            
            data = next(self.loader)
        except StopIteration:
            print('raise StopIteration Exception....')
             # epoch is down
            if self.chunk_idx == len(self.dataset.chunks)-1:
                self.chunk_idx = 0
                self._init_loader(first_iter=False)
                raise StopIteration
            else:
                # data in the current chunk have been consumed
                self.chunk_idx += 1
                self.clear()
                self.dataset.load_data(self.chunk_idx)
                self._init_loader(first_iter=True)
                data = next(self.loader)
        return data