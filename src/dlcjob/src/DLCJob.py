import os
import json
import math
import random
import numpy as np
import zmq
from multiprocessing import Process, Queue
from typing import Optional, Union, Sequence, Iterable, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t


jobinfo = "/share/{}.json".format(os.environ.get('JOBNAME'))
init_channel = 'ipc:///share/init.ipc'
ipc_channel = 'ipc:///share/runtime.ipc'


class DLCJobDataset(Dataset):
    def __init__(self, dtype='train'):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__process__`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~DLCJobDataLoader`.
        
        .. note::
        Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self.samples` as Map<Key, Value>.
        Overwriting meth:`__process__` allows you to replace var:`self.samples` and var:`self.targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            dtype: dataset type, train/validation/test for supervised and unsupervised training/testing
        """
        if dtype not in ["train", "validation", "test", 
                         "train/samples", "validation/samples", "test/samples"]:
            raise ValueError("invalid dataset type".format(dtype))
        
        self.dtype = dtype.split('/')
        while not os.path.exists(jobinfo): pass
        with open(jobinfo, 'rb') as f:
            job_meta = json.load(f)

        self.bucket = job_meta['datasource']['bucket']
        self.lazy = job_meta['qos']['LazyLoading']
        self.usecache = job_meta['qos']['UseCache']
        self.maxpart = job_meta['qos']['MaxPartMill'] * 1e6
        if self.maxpart == 0:
            self.maxpart = math.inf
        
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
            
            mfst_etags = self.job_info["ChunkETags"][self.dtype[0]]["manifests"]
            self.manifest = None
            for mfst_etag in mfst_etags:
                mfst_chunk = self.dataset_col.find_one({"ETag": mfst_etag})
                mfst_fpath = "/{}/{}".format(mfst_chunk['Location'], mfst_etag)
                if self.manifest is None:
                    self.manifest = pd.read_csv(manifest)
                else:
                    self.manifest = pd.concat([self.manifest, pd.read_csv(manifest)])
            self.loadChunksFromDLCache()
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
            self.loadChunksFromCloud()
        
        self.nfsFilePaths = []
        self.load_data(chunk_idx=0)
        
        self.miss_queue = Queue()
        self.unused_idx = set(list(range(self.__len__())))

    def try_get_item(self, idx):
        if self.lazy:
            if len(self.unused_idx) == 0:
                self.unused_idx = set(list(range(self.__len__())))
            
            sample_path, target_path = self.samples[idx], self.targets[idx] if self.targets else None                
            if self.usecache:
                def helper(path, reader):
                    if path is None:
                        return None
                    tmpfs_path = '/runtime{}'.format(path)
                    
                    read_idx = idx
                    read_val = None
                    
                    # 3-level data hit
                    if os.path.exists(tmpfs_path):
                        read_val = reader(tmpfs_path)
                        # print("read tmpfs file {}".format(tmpfs_path))
                    elif os.path.exists(path):
                        read_val = reader(path)
                        # print("miss tmpfs file {}".format(tmpfs_path))
                    else:
                        # Substitutable cache hit
                        # 1. randomly select an unused data point
                        # 2. replace the idx with sub_idx
                        # 3. socket in data loader notify client to update data access sequence
                        print('miss nfs file {}'.format(path))
                        while True:
                            if len(self.unused_idx) == 0:
                                sub_idx = random.randint(0, self.__len__()-1)
                            else:
                                sub_idx = random.choice(list(self.unused_idx))
                                                            
                            sub_path = self.samples[sub_idx]
                            if os.path.exists(sub_path):
                                etag = path.split('/')[1]
                                self.miss_queue.put([[idx, etag], sub_idx])
                                read_idx = sub_idx
                                break
                            else:
                                sub_etag = sub_path.split("/")[1]
                                self.miss_queue.put([[sub_idx, sub_etag], sub_idx])
                        
                        read_val = self.try_get_item(sub_idx)
                    
                    if read_idx is not None and read_idx in self.unused_idx:
                        self.unused_idx.remove(read_idx)
                    return read_val
                
                sample = helper(sample_path, self.__sample_reader__)
                target = helper(target_path, self.__target_reader__) if len(self.dtype) == 1 else None
            else:
                sample = self.client.get_object(Bucket=self.bucket, Key=sample_path)['Body'].read()
                target = self.client.get_object(Bucket=self.bucket, Key=target_path)['Body'].read() if len(self.dtype) == 1 else None
            return sample, target
        else:
            return self.samples[idx], self.targets[idx] if self.targets else None

    def loadChunksFromDLCache(self):
        """Load all chunks (MongoDB query result) of the given dataset. 
        
        In the LazyLoading mode, there is only 1 group because no memory pressure.
        Otherwise, we group chunks to make them fit the MaxPartMill constraint.

        Returns:
            None: initialize the self.sample_chunks and self.target_chunks
        """
        chunk_etags = self.job_info['ChunkETags'][self.dtype[0]]
        
        def helper(target_etags):
            chunk_groups = []
            chunks = self.dataset_col.find({"ChunkETag": {"$in": target_etags}})
            total_size = 0
            chunk_groups.append([])
            for chunk in chunks:
                total_size += int(chunk['Size'])
                if total_size >= self.maxpart:
                    chunk_groups.append([])
                    total_size = 0
                elif int(chunk['Size']) > self.maxpart:
                    raise Exception('File {} size is greater than assigned maxpartory.'.format(chunk['Key']))
                else:
                    chunk_groups[-1].append(chunk)
            return chunk_groups
    
        self.sample_chunks = helper(chunk_etags['samples'])
        if len(self.dtype) == 1:
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

        keys = self.job_info['Datasource']['keys'][self.dtype[0]]
        
        def helper(target_keys):
            total_size = 0
            chunk_groups = []
            pages = load_pages(target_keys)
            for i in range(len(pages)):
                for j in range(len(pages[i]['Contents'])):
                    dataobj = pages[i]["Contents"][j]
                    s = int(dataobj['Size'])
                    total_size += s
                    if total_size > self.maxpart:
                        chunk_groups.append([])
                        total_size = 0
                    elif s > self.maxpart:
                        raise Exception('File {} size is greater than assigned maxpartory.'.format(dataobj['Key']))
                    else:
                        chunk_groups[-1].append(dataobj)
            return chunk_groups
        
        self.sample_chunks = helper(keys['samples'])
        if len(self.dtype) == 1:
            self.target_chunks = helper(keys['targets'])  

    def load_data(self, partition_index):
        """Load file paths or actual data in the given partition
        
        Initialize the self.samples and self.targets, where self.samples is a dict with format {key: file_path/data}
        user performs data (X and y) processing in the __process__ function, that convert self.samples ans self.targets
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
            nfs_path = []
            for chunk in chunks[partition_index]:
                key, loc = chunk['Key'], chunk['Location']
                if self.usecache:
                    if not chunk:
                        nfs_path.append(None)
                        continue
                    etag = chunk['ChunkETag']
                    chunk_path = '/{}/{}'.format(loc, etag)
                    # decompressed folder, so key has the .tar.gz extension
                    if os.path.isdir(chunk_path):
                        # We assume data won't be immediately deleted after being downloaded by Manager.
                        for root, dirs, files in os.walk(chunk_path):
                            for name in files:
                                fpath = os.path.join(root, name)
                                nfs_path.append(fpath)
                                # dummy_key = chunk key + '/' + sample file name
                                dummy_key = os.path.join(key, fpath.replace(chunk_path, ''))
                                data[dummy_key] = fpath if self.lazy else reader(fpath)
                    else:
                        nfs_path.append(chunk_path)
                        data[key] = chunk_path if self.lazy else reader(chunk_path)
                else:
                    data[key] = key
                    nfs_path.append(None)
            return data, nfs_path
            
        self.samples, sample_fpaths = helper(self.sample_chunks, self.__sample_reader__)
        self.targets, target_fpaths = helper(self.target_chunks, self.__target_reader__)
        
        # build mappings between samples and targets if the dataset is file-based
        # and the manifest is specified
        if self.lazy and self.targets and self.manifest:
            samples_ = {}
            targets_ = {}
            sample_fpaths_ = []
            target_fpaths = []
            for _, row in self.manifest.iterrows():
                dummy_key = os.path.join(row['sample_chunk'], row['sample'])
                samples_[dummy_key] = self.samples[dummy_key]
                dummy_key = os.path.join(row['target_chunk'], row['target'])
                targets_[dummy_key] = self.targets[dummy_key]
            self.samples = samples_
            self.targets = targets_
            sample_fpaths = list(self.samples.values())
            target_fpaths = list(self.targets.values())
        
        self.nfsFilePaths = list(zip(sample_fpaths, target_fpaths))
        self.samples, self.targets = self.__process__()

    def __sample_reader__(self, path: str = None, raw_bytes: bytes = None):
        """this function defines the logic of reading sample data (X) from a file

        Args:
            path (string): read from a file
            raw_bytes (bytes): read from raw bytes
            
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __target_reader__(self, path: str = None, raw_bytes: bytes = None):
        """this function defines the logic of reading target data (Y) from a file

        Args:
            path (string): read from a file
            raw_bytes (bytes): read from raw bytes

        Raises:
            NotImplementedError: _description_
        """
        pass
    
    def __process__(self) -> Tuple[List, List]:
        """process self.samples ans self.target

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
        self.num_batches = math.ceil(len(self.dataset.samples)/batch_size)
        
        self.lazy = self.dataset.qos['LazyLoading']
        self.partition_index = 0
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
                sub_idx = info[1]
                msg = "{}:{} {}".format(miss_idx, miss_etag, sub_idx)
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
            if self.partition_index == len(self.dataset.sample_chunks)-1:
                self.partition_index = 0
                self._init_loader(first_iter=False)
                raise StopIteration
            else:
                # data in the current chunk have been consumed
                self.partition_index += 1
                self.clear()
                self.dataset.load_data(self.partition_index)
                self._init_loader(first_iter=True)
                data = next(self.loader)
        return data