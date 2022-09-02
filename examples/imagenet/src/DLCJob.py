import os
import json
import concurrent.futures
import multiprocessing
from math import inf
import math
import threading
import numpy as np
import time
from typing import Optional, Union, Sequence, Iterable, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

        
class DLCJobDataset(Dataset):
    def __init__(self, keys: List[str] = None, shuffle=False):
        # """An abstract class subclassing the torch.utils.data.Dataset class
        
        # All datasets that represent a map from keys to data samples should subclass
        # it. All subclasses should overwrite :meth:`__convert__`, supporting pre-processing loaded data. 
        # Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        # data sample for a given key. Subclasses could also optionally overwrite
        # :meth:`__len__`, which is expected to return the size of the dataset by many
        # :class:`~torch.utils.data.Sampler` implementations and the default options
        # of :class:`~DLCJobDataLoader`.
        
        # .. note::
        # Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self.data` as Map<Key, Value>.
        # Overwriting meth:`__convert__` allows you to replace var:`self.data` and var:`self.targets` with
        # iteratable variables that can be iterated in meth:`__get_item__`.
        
        # Args:
        #     keys (List, optional): a list of bucket keys. Defaults to None, meaning loading all keys in the bucket.
        #     shuffle (Bool, optional): default False
        # """
        self.shuffle = shuffle
        jobname = os.environ.get('JOBNAME')
        self.keys = keys
        with open('/jobsmeta/{}.json'.format(jobname), 'r') as f:
            self.job = json.load(f)
        self.qos = self.job['qos']
        self.bucket = self.job['dataSource']['bucket']
        self.chunks = []
        
        if self.qos['UseCache']:
            from pymongo.mongo_client import MongoClient
            while True:
                try:
                    with open("/share/{}.json".format(jobname), 'rb') as f:
                        resp = dotdict(json.load(f))
                        break
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    pass
            mongo_client = MongoClient(resp.mongoUri)
            self.jobId = resp.jobId
            self.job_col = mongo_client.Cacher.Job
            self.dataset_col = mongo_client.Cacher.Datasets
            self.load_cache_keys()
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
            self.load_s3_keys()

        self.nfsFilePaths = []
        self.targets = None
        self.load_data(0)
                    
    def get_data(self, index):
        if self.qos['LazyLoading']:
            return self.read(self.data[index])
        else:
            return self.data[index]
    
    def get_target(self, index):
        return self.targets[index]

    def load_cache_keys(self):
        maxmem = self.qos['MaxMemoryMill']*1e6  
        etags = self.job_col.find_one({"Meta.JobId": self.jobId})['ETags']
        chunks = [chunk for chunk in self.dataset_col.find({"ChunkETag": {"$in": etags}})]
        
        if self.shuffle:
            np.random.shuffle(etags)
                    
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
        
    def load_s3_keys(self):
        paginator = self.client.get_paginator('list_objects_v2')
        if self.keys is not None:
            pages = []
            for k in self.keys:
                pages.extend(paginator.paginate(Bucket=self.bucket, Prefix=k))
        else:
            pages = paginator.paginate(Bucket=self.bucket)
        
        maxmem = self.qos['MaxMemoryMill']*1e6
        maxmem = inf if maxmem==0 else maxmem
        total_size = 0
        self.chunks.append([])
        if self.shuffle:
            np.random.shuffle(pages)
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
    
    def read(self, chunk):
        key = chunk['Key']
        if self.qos['UseCache']:
            etag = chunk['ChunkETag']
            loc = chunk['Location']
            nfs_path = '/{}/{}'.format(loc, etag)
            tmpfs_path = '/runtime{}'.format(nfs_path)
            try:
                # while not os.path.exists(tmpfs_path): pass
                with open(tmpfs_path, 'rb') as f:
                    val = f.read()
                threading.Thread(target=lambda: os.remove(tmpfs_path), daemon=True).start()  # 在后台删除
            except FileNotFoundError:
                print("miss file {}".format(tmpfs_path))
                with open(nfs_path, 'rb') as f:
                    val = f.read()
            else:
                print("miss file {}".format(nfs_path))
                with open('/share/datamiss', 'w') as f:
                    f.writelines("{}:{}".format(self.bucket, key))
                while not os.path.exists(etag): pass
                with open(nfs_path, 'rb') as f:
                    val = f.read()
        else:
            val = self.client.get_object(Bucket=self.bucket, Key=key)['Body'].read()
        return val

    def load_data(self, index):
        self.data = {}
        self.keys = []        
        if self.qos['LazyLoading']:
            # LazyLoading mode fits dataset which data items 
            # are individual files, such as image dataset
            for chunk in self.chunks[index]:
                self.keys.append(chunk['Key'])
                self.nfsFilePaths.append('/{}/{}'.format(chunk['Location'], chunk['ChunkETag']))
                self.data[chunk['Key']] = chunk
        else:
            # Normal mode fits tabular or block datasets, 
            # so we load a subset of the original dataset here
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for chunk in self.chunks[index]:
                    futures.append(executor.submit(self.read, chunk))
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    chunk = self.chunks[index][i]
                    self.keys.append(chunk['Key'])
                    self.nfsFilePaths.append('/{}/{}'.format(chunk['Location'], chunk['ChunkETag']))
                    self.data[chunk['Key']] = future.result()
                
        self.data, self.targets = self.__convert__()
    
    def __convert__(self) -> Tuple[List, List]:
        """convert self.data

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        # """
        # Args:
        #     index (int): Index

        # Returns:
        #     (Any): Sample and meta data, optionally transformed by the respective transforms.
        # """
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError
    
    
class DLCJobDataLoader(object):
    def __init__(self, dataset: DLCJobDataset, 
                 batch_size: Optional[int] = 1, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        # """
        # Data loader. Combines a dataset and a sampler, and provides an iterable over
        # the given dataset.

        # The :class:`~torch.utils.data.DataLoader` supports both map-style and
        # iterable-style datasets with single- or multi-process loading, customizing
        # loading order and optional automatic batching (collation) and memory pinning.

        # See :py:mod:`torch.utils.data` documentation page for more details.

        # Args:
        #     dataset (DLCJobDataset): dataset from which to load the data.
        #     batch_size (int, optional): how many samples per batch to load
        #         (default: ``1``).
        #     sampler (Sampler or Iterable, optional): defines the strategy to draw
        #         samples from the dataset. Can be any ``Iterable`` with ``__len__``
        #         implemented. If specified, :attr:`shuffle` must not be specified.
        #     batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
        #         returns a batch of indices at a time. Mutually exclusive with
        #         :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
        #         and :attr:`drop_last`.
        #     num_workers (int, optional): how many subprocesses to use for data
        #         loading. ``0`` means that the data will be loaded in the main process.
        #         (default: ``0``)
        #     collate_fn (callable, optional): merges a list of samples to form a
        #         mini-batch of Tensor(s).  Used when using batched loading from a
        #         map-style dataset.
        #     pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
        #         into CUDA pinned memory before returning them.  If your data elements
        #         are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        #         see the example below.
        #     drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        #         if the dataset size is not divisible by the batch size. If ``False`` and
        #         the size of dataset is not divisible by the batch size, then the last batch
        #         will be smaller. (default: ``False``)
        #     timeout (numeric, optional): if positive, the timeout value for collecting a batch
        #         from workers. Should always be non-negative. (default: ``0``)
        #     worker_init_fn (callable, optional): If not ``None``, this will be called on each
        #         worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        #         input, after seeding and before data loading. (default: ``None``)
        #     generator (torch.Generator, optional): If not ``None``, this RNG will be used
        #         by RandomSampler to generate random self.indexes and multiprocessing to generate
        #         `base_seed` for workers. (default: ``None``)
        #     prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
        #         in advance by each worker. ``2`` means there will be a total of
        #         2 * num_workers samples prefetched across all workers. (default: ``2``)
        #     persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
        #         the worker processes after a dataset has been consumed once. This allows to
        #         maintain the workers `Dataset` instances alive. (default: ``False``)


        # .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
        #             cannot be an unpicklable object, e.g., a lambda function. See
        #             :ref:`multiprocessing-best-practices` on more details related
        #             to multiprocessing in PyTorch.

        # .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
        #             When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
        #             it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
        #             rounding depending on :attr:`drop_last`, regardless of multi-process loading
        #             configurations. This represents the best guess PyTorch can make because PyTorch
        #             trusts user :attr:`dataset` code in correctly handling multi-process
        #             loading to avoid duplicate data.

        #             However, if sharding results in multiple workers having incomplete last batches,
        #             this estimate can still be inaccurate, because (1) an otherwise complete batch can
        #             be broken into multiple ones and (2) more than one batch worth of samples can be
        #             dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
        #             cases in general.

        #             See `Dataset Types`_ for more details on these two types of datasets and how
        #             :class:`~torch.utils.data.IterableDataset` interacts with
        #             `Multi-process data loading`_.

        # .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
        #             :ref:`data-loading-randomness` notes for random seed related questions.
        # """
    
        self.dataset = dataset
        self.batch_size = batch_size
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
        self.num_batches = math.ceil(len(self.dataset.data)/self.batch_size)
        self.lazy = self.dataset.qos['LazyLoading']
        self.init_loader()
        self.index = 1
        
        # prefetch size depends on the chunk size when LazyLoading is disabled
        if not self.lazy:
            with open('/share/prefetchKeys.json', 'w') as f:
                json.dump(f, {
                    "meta": {'num_workers': self.num_workers, 'batch_size': self.batch_size, 'LazyLoading': self.lazy}, 
                    "policy": self.dataset.nfsFilePaths})
    
    def init_loader(self):
        # shuffle if disabled under the LazyLoading mode
        loader = DataLoader(self.dataset, self.batch_size, not self.lazy, self.sampler, self.batch_sampler, self.num_workers, self.collate_fn, 
                            self.pin_memory, self.drop_last, self.timeout, self.worker_init_fn, self.multiprocessing_context, self.generator, 
                            prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
        self.loader = loader._get_iterator()
        file_paths = np.array(self.dataset.nfsFilePaths)
        if self.lazy:
            prefetchPaths = [file_paths[indices].tolist() for indices in loader._index_sampler]
            with open('/share/prefetchKeys.json', 'w') as f:
                json.dump(
                    {
                        "meta": {'num_workers': self.num_workers, 'batch_size': self.batch_size, 'LazyLoading': self.lazy}, 
                        "policy": prefetchPaths
                    }, f)
        else:
            with open('/share/next', 'w') as f:
                f.write(str(time.time()))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = self.loader.next()
            if self.lazy:
                with open('/share/next', 'w') as f:
                    f.write(str(time.time()))
        except StopIteration:
            if self.index == len(self.dataset.chunks):  # epoch is down
                self.index = 1
                raise StopIteration
            else:
                self.dataset.load_data(self.index)
                self.init_loader()
                data = self.loader.next()
                self.index += 1
        return data