import os
import json
import bson
import random
import numpy as np
import time
from datetime import datetime
import grpc
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
import threading
import queue
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Any,
    Union,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from collections import defaultdict
import itertools
from scipy.special import softmax
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import _utils
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t
import torch.utils.data._utils.worker as worker
import torch.utils.data._utils.signal_handling as signal_handling
from pymongo.mongo_client import MongoClient
from utils import *
warnings.filterwarnings("ignore")

COOL_DOWN_SEC = 600
cpu_count = multiprocessing.cpu_count()
cores = np.arange(1, cpu_count+1)
class CHUNK_STATUS:
    PREPARE = 0
    ACTIVE = 1
    PENDING = 2
    COOL_DOWN = 3
    INACTIVE = 4


T_co = TypeVar('T_co', covariant=True)


class DLCJobDataset(Generic[T_co]):
    def __init__(self, dataset_type='train'):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`process`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~DLCJobDataLoader`.
        
        .. note::
        Subclassing ~DLCJobDataset will load data under provided keys from DLCache to var:`self._samples` as Map<Key, Value>.
        Overwriting meth:`process` allows you to replace var:`self._samples` and var:`self._targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            dataset_type: dataset type, train/validation/test for supervised and unsupervised training/testing
        """
        if dataset_type not in ["train", "validation", "test"]:
            raise ValueError("invalid dataset type".format(dataset_type))
        
        self.dataset_type = dataset_type
        jobinfo = "/share/{}.json".format(os.environ.get('JOBNAME'))
        while not os.path.exists(jobinfo): pass
        with open(jobinfo, 'rb') as f:
            job_meta = json.load(f)

        self.lazy = job_meta['qos']['LazyLoading']
        
        # manifest files for mapping object path from cloud to local
        self.__samples_manifest = {}
        self.__targets_manifest = {}

        mongo_client = MongoClient(job_meta['mongoUri'])
        self.db = mongo_client.Cacher
        cursor = self.db.Job.find_one({"Meta.JobId": job_meta['jobId']})
        self.job_info = {"ChunkETags": cursor["ChunkETags"], "Meta": cursor["Meta"]}
        
        self.__build_manifests()
        self.num_partitions = 1 if self.lazy else len(self.samples_manifest)
        
        self._load_partition_data()
        self._miss_queue = queue.Queue()
        self._handle_miss_thread = threading.Thread(target=self._handle_miss, daemon=True)
        self._handle_miss_thread.start()

    def _handle_miss(self):
        # grpc consumes high memory
        manager_uri = "dlcpod-manager:50051"
        while True:
            miss_etag = self._miss_queue.get(block=True)
            with grpc.insecure_channel(manager_uri) as channel:
                cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
                stub = pb_grpc.DataMissStub(channel)
                stub.call(pb.DataMissRequest(cred=cred, etag=miss_etag))
            del cred, stub, miss_etag
    
    @property
    def samples_manifest(self):
        return self.__samples_manifest
    
    @property
    def targets_manifest(self):
        return self.__targets_manifest
    
    def __build_manifests(self):
        """Load all chunks (MongoDB query result) of the given dataset. 
        
        In the LazyLoading mode, there is only 1 group because no memory pressure.
        Otherwise, we group chunks to make them fit the MaxPartMill constraint.

        Returns:
            None: initialize the self._sample_chunks and self._target_chunks
        """
        def load(etags):
            data = {}
            if etags:
                chunks_iter = self.db.Datasets.aggregate([
                    {"$match": {"ChunkETag": {"$in": etags}}},
                    {"$project": {"Key": 1, "Location": 1, "ChunkETag": 1, "_id": 0}}
                ])
                for chunk in chunks_iter:
                    cloud_path, loc, chunk_etag = chunk['Key'], chunk["Location"], chunk['ChunkETag']
                    local_path = '/{}/{}'.format(loc, chunk_etag)
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

        chunk_etags = self.job_info['ChunkETags'][self.dataset_type]
        self.__samples_manifest = load(chunk_etags['samples'])
        self.__targets_manifest = load(chunk_etags['targets'])
        self.chunk_etags = [*chunk_etags['samples'], *chunk_etags["targets"]]
    
    def _load_partition_data(self, partition_idx=0):
        if self.lazy:
            self._process(list(self.samples_manifest.keys()), list(self.targets_manifest.keys()))
        else:
            self._process([list(self.samples_manifest.keys())[partition_idx]], [list(self.targets_manifest.keys())[partition_idx]])
    
    def _process(self, sample_files: List[str], target_files: List[str]=None):
        r"""Given the cloud keys of input files,
        you need to use the samples_manifest and targets_manifest to 
        generate X, Y that can be iterated in the __getItem__ function.
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getItem__(self, index: int):     
        r"""get the sample and target at the given index
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """       
        raise NotImplementedError
            
    def __getitem__(self, index: int):
        if self.lazy:
            try:
                return self.__getItem__(index)
            except FileNotFoundError as ex:
                miss_etag = ex.filename.split('/')[-1]
                self._miss_queue.put(miss_etag)
                sub_idx = random.randint(index+1, len(self) - 1)
                return self.__getItem__(sub_idx)
        else:
            return self.__getItem__(index)
    
    def __len__(self) -> int:
        raise NotImplementedError


class DLCJobDataLoader(DataLoader):
    def __init__(self, dataset: DLCJobDataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2, persistent_workers: bool = False, autoscale_workers: bool = True,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)
        
        assert self.num_workers > 0
        assert self.prefetch_factor > 0
        
        self.check_worker_number_rationality()
        self.num_batches = len(self)
        self.mongo_operation_queue = queue.Queue()
        self.cool_down_proc = None
        self.autoscale_workers = autoscale_workers
        threading.Thread(target=self.async_mongo_operator, daemon=True).start()
        self.init_tunner()
    
    def init_tunner(self):
        self.tune_iters = 0
        
        self.tune_freqs = []
        k = 2*cpu_count
        w = cpu_count
        while w > 0:
            self.tune_freqs.extend([k] * w)
            k *= 2
            w = w//2
        
        self.tune_freqs = itertools.cycle(self.tune_freqs)
        self.next_tune_freq = next(self.tune_freqs)
        self.load_time_cache = defaultdict(list)  # the latest `num_batches` # of load time
        self.worker_weights = None
        self.perf_metrics = defaultdict(float)
        self.avg_load_time_history = defaultdict(list)
        
    def async_mongo_operator(self):
        try:
            while True:
                collection, func, opertion = self.mongo_operation_queue.get(block=True)
                if func == 'update_many':
                    self.dataset.db[collection].update_many(**opertion)
                del collection, func, opertion
        except KeyboardInterrupt:
            pass
           
    def expire_chunks(self):
        time.sleep(COOL_DOWN_SEC)   
        self.mongo_operation_queue.put(("Datasets", "update_many", 
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
        
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.cool_down_proc is not None:
            if self.cool_down_proc.is_alive():
                self.cool_down_proc.terminate()
                self.cool_down_proc.join()
            del self.cool_down_proc
        self.cool_down_proc = multiprocessing.Process(target=self.expire_chunks)
        
        if self.autoscale_workers and self._iterator is not None:
            self.tune_iters = self._iterator._tune_iters
            self.load_time_cache = self._iterator._load_time_cache
            self.worker_weights = self._iterator._worker_weights
            self.perf_metrics = self._iterator._perf_metrics
            self.avg_load_time_history = self._iterator._avg_load_time_history
            num_workers = self._iterator._active_workers.value
            next_tune_freq = self._iterator._next_tune_freq
        else:
            num_workers = self.num_workers
            next_tune_freq = self.next_tune_freq
            
        self._iterator = _DLCJobDataLoaderIter(self, self.num_batches, num_workers, self.cool_down_proc, 
                                               self.mongo_operation_queue, self.autoscale_workers, self.load_time_cache, self.worker_weights, 
                                               self.perf_metrics, self.avg_load_time_history, self.tune_iters, self.tune_freqs, next_tune_freq)

        return self._iterator
        

class StatefulCycleIterator:        
    def __init__(self, num_workers=0):
        self.init_num_workers = num_workers
        self._workers_status = [1 for _ in range(num_workers)]
        self._ptr = 0
    
    def __next__(self):
        for _ in range(len(self._workers_status)):
            if self._ptr >= len(self._workers_status):
                self._ptr = 0
            if self._workers_status[self._ptr] == 1:
                w = self._ptr
                self._ptr += 1
                return w
            else:
                self._ptr += 1
        return None

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self._workers_status)
    
    def get_ptr(self):
        return self._ptr
    
    def set_ptr(self, pos):
        self._ptr = pos
    
    # append to the end
    def append(self, worker_id, status=1):
        assert worker_id == len(self._workers_status)
        self._workers_status.append(status)
    
    def set_status(self, index, status):
        assert index < len(self._workers_status)
        self._workers_status[index] = status
    
    def get_status(self, index):
        return self._workers_status[index]
    
    def reactive_worker(self):
        for i in range(len(self)):
            if self._workers_status[i] == 0:
                self._workers_status[i] = 1
                return True
        return False
    
    def deactive_worker(self):
        for i in range(len(self)):
            if self._workers_status[i] == 1:
                self._workers_status[i] = 0
                return True
        return False
    
    def reset(self):
        self._ptr = 0
        
        
class _DLCJobDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader, num_batches, opt_num_workers,
                 cool_down_proc: multiprocessing.Process = None, mongo_operation_queue: queue.Queue = None, autoscale_workers: bool = True,
                 load_time_cache: defaultdict = None, worker_weights: np.array = None, perf_metrics: np.array = None,
                 avg_load_time_history: np.array = None, tune_iters: int = None, tune_freqs: itertools.cycle = None,
                 next_tune_freq: int = None):
        super(_DLCJobDataLoaderIter, self).__init__(loader)
        
        self._num_batches = num_batches
        self._num_workers = opt_num_workers
        self._autoscale_workers = autoscale_workers
        self._prefetch_factor = loader.prefetch_factor
        self._tune_freqs = tune_freqs
        self._next_tune_freq = next_tune_freq
        self.lazy = self._dataset.lazy

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context
        self._multiprocessing_context = multiprocessing_context
        self._worker_init_fn = loader.worker_init_fn

        # We don't consider DataPipe currently
        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        # if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
        #     self._worker_init_fn = functools.partial(_sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)
                
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        manager = multiprocessing.Manager()
        self._active_workers = manager.Value('_active_workers', 0)
        self._index_queues = []
        self._workers = []
        self._worker_queue_idx_cycle = None
        for i in range(self._num_workers):
            self._spawn_worker(i)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_DLCJobDataLoaderIter._clean_up_worker, w)

        self._partition_idx = 0
        
        # .pid can be None only before process is spawned (not the case, so ignore)
        signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        
        self._mongo_operation_queue = mongo_operation_queue
        self._cool_down_proc = cool_down_proc
        
        self._tune_iters = tune_iters
        self._load_time_cache = load_time_cache
        self._worker_weights = worker_weights
        self._perf_metrics = perf_metrics
        self._avg_load_time_history = avg_load_time_history
        self._reset(loader, first_iter=True)
        
    def _reset(self, loader, first_iter=False):    
        super()._reset(loader, first_iter)
        
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)

        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._reorder_dict = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in _reorder_dict.values() if len(v) == 1)

        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = StatefulCycleIterator(num_workers=self._active_workers.value)
        
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._active_workers.value):
                self._index_queues[idx].put(worker._ResumeIteration(self._shared_seed))
            resume_iteration_cnt = self._active_workers.value
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
                    
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._active_workers.value):
            self._try_put_index()

    def _spawn_worker(self, worker_id):
        if self._worker_queue_idx_cycle is not None and self._worker_queue_idx_cycle.reactive_worker():
            self._active_workers.value += 1
            return

        idx_queue = self._multiprocessing_context.Queue()
        idx_queue.cancel_join_thread()
        w = self._multiprocessing_context.Process(target=worker._worker_loop,
                                                 args=(self._dataset_kind, self._dataset, idx_queue, self._worker_result_queue, self._workers_done_event,
                                                          self._auto_collation, self._collate_fn, self._drop_last, self._base_seed, self._worker_init_fn,
                                                          worker_id, self._active_workers, self._persistent_workers, self._shared_seed))
        w.daemon = True
        w.start()
        self._index_queues.append(idx_queue)
        self._workers.append(w)
        if self._worker_queue_idx_cycle is not None:
            self._worker_queue_idx_cycle.append(worker_id)
        self._active_workers.value += 1
    
    def _pause_worker(self):
        if self._worker_queue_idx_cycle.deactive_worker():
            self._active_workers.value -= 1
    
    def _reset_tunner(self):
        self._worker_weights = None
        self._perf_metrics = defaultdict(float)
        self._load_time_cache = defaultdict(list)  # the latest `num_batches` # of load time
        self._avg_load_time_history = defaultdict(list)
    
    def _tune_worker_num(self):
        mean = np.mean
        num_workers = self._active_workers.value
        
        # buffer `num_batches` load time measurements
        if len(self._load_time_cache[num_workers]) > self._next_tune_freq:
            self._load_time_cache[num_workers].clear()

        if self._rcvd_idx == 1 or (self._rcvd_idx % self._next_tune_freq == 0):
            # print(self._worker_weights)
            if len(self._load_time_cache[num_workers]) == 0:
                return
            
            if len(self._load_time_cache[num_workers]) > 0:
                self._avg_load_time_history[num_workers].append(mean(self._load_time_cache[num_workers]))
            
            # update weights
            if num_workers in self._perf_metrics:
                # due to the measurement jitter, we use the alpha to balance historical and the latest performance measurement 
                alpha = 0.8
                self._perf_metrics[num_workers] = alpha * self._perf_metrics[num_workers] + (1-alpha) * mean(self._load_time_cache[num_workers])
            else:
                self._perf_metrics[num_workers] = mean(self._load_time_cache[num_workers])
            if len(self._perf_metrics) == cpu_count:
                self._worker_weights = softmax( 1/np.array(list(self._perf_metrics.values())) )
            
            # get the next `num_worker` value to test
            if self._worker_weights is not None:
                new_num_workers = np.random.choice(cores, size=1, replace=False, p=self._worker_weights)[0]
            else:
                new_num_workers = cores[self._tune_iters]
            
            delta = new_num_workers - num_workers
            for _ in range(abs(delta)):
                if delta > 0:
                    self._spawn_worker(worker_id=self._active_workers.value)
                elif delta < 0:
                    self._pause_worker()
            
            if delta > 0:
                pos = self._worker_queue_idx_cycle.get_ptr()
                for _ in range(self._prefetch_factor):
                    self._worker_queue_idx_cycle.set_ptr(pos)
                    for _ in range(delta):
                        self._try_put_index()
                    
            if self._perf_metrics[num_workers] is not None:
                self._tune_iters += 1
            self._next_tune_freq = next(self._tune_freqs)
            print('change worker num to: {}'.format(self._active_workers.value))

    def _try_put_index(self):    
        try:
            batched_idxs = self._next_index()
        except StopIteration:
            return

        worker_queue_idx = next(self._worker_queue_idx_cycle)
        if worker_queue_idx is None:
            return
        
        self._index_queues[worker_queue_idx].put((self._send_idx, batched_idxs))
        self._reorder_dict[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
    
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._worker_queue_idx_cycle.get_status(worker_id) != -1 and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise
        
    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data
    
    def _process_data(self, data):    
        self._rcvd_idx += 1
        if isinstance(data, worker.ExceptionWrapper):
            data.reraise()
        self._try_put_index()
        return data

    def _next_data(self):
        start = time.time()

        if self._rcvd_idx == 0:
            # update chunk status to ACTIVE
            now = datetime.utcnow().timestamp()
            self._mongo_operation_queue.put(('Datasets', 'update_many', 
                                            {
                                                "filter": {"ChunkETag": {"$in": self._dataset.chunk_etags}}, 
                                                "update": {
                                                    "$set": { "Status.code": CHUNK_STATUS.ACTIVE},
                                                    "$inc": {"Status.active_count": 1},
                                                    "$push": {"References": bson.timestamp.Timestamp(int(now), inc=1)}}
                                            }))
            
        while True:
            try:
                while self._rcvd_idx < self._send_idx:
                    info = self._reorder_dict[self._rcvd_idx]
                    worker_id = info[0]
                    if len(info) == 2 or self._worker_queue_idx_cycle.get_status(worker_id) != -1:  # has data or is still active
                        break
                    del self._reorder_dict[self._rcvd_idx]
                    self._rcvd_idx += 1
                else:
                    if not self._persistent_workers:
                        self._shutdown_workers()
                    raise StopIteration

                # Now `self._rcvd_idx` is the batch index we want to fetch
                # Check if the next sample has already been generated
                if len(self._reorder_dict[self._rcvd_idx]) == 2:
                    data, _active_workers = self._reorder_dict.pop(self._rcvd_idx)[1]
                    data = self._process_data(data)
                    break
                
                assert not self._shutdown and self._tasks_outstanding > 0
                idx, data, _active_workers  = self._get_data()
                self._tasks_outstanding -= 1

                if idx != self._rcvd_idx:
                    # store out-of-order samples
                    self._reorder_dict[idx] += ((data, _active_workers),)
                else:
                    del self._reorder_dict[idx]
                    data = self._process_data(data)
                    break
            except StopIteration:
                # epoch is down
                if self._partition_idx == self._dataset.num_partitions-1:
                    self._partition_idx = 0
                    
                    # update chunk status to COOL_DOWN
                    now = datetime.utcnow().timestamp()
                    self._mongo_operation_queue.put(("Datasets", "update_many", 
                                                    {
                                                        "filter": {
                                                            "ChunkETag": {"$in": self._dataset.chunk_etags},
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
                                                            "ChunkETag": {"$in": self._dataset.chunk_etags},
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
                    self._cool_down_proc.start()
                    raise StopIteration
                else:                    
                    # data in the current part have been consumed    
                    self._partition_idx += 1
                    self._dataset._load_partition_data(self._partition_idx)
                    continue
                
        # ensure the `num_workers` is consensus while reading the batch
        # we skip the first batch because the reset function needs to prefetch data synchronously
        if _active_workers is not None:
            self._load_time_cache[_active_workers].append(time.time() - start)
        
        if self._autoscale_workers:
            self._tune_worker_num()
            
        return data
        
    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._worker_queue_idx_cycle.get_status(worker_id) != -1 or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._worker_queue_idx_cycle.set_status(worker_id, -1)

        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._worker_queue_idx_cycle.get_status(worker_id) != -1:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()
