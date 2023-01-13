import os
import json
import math
import random
import numpy as np
import time
import grpc
import databus.dbus_pb2 as pb
import databus.dbus_pb2_grpc as pb_grpc
import threading
import queue
import pickle
import shutil
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
import zmq
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import _utils
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t
import torch.utils.data._utils.worker as worker
import torch.utils.data._utils.signal_handling as signal_handling
from lib.utils import *
import warnings
warnings.filterwarnings("ignore")

cpu_count = multiprocessing.cpu_count() - 4
cores = np.arange(1, cpu_count+1)
init_channel = 'ipc:///share/init.ipc'
ipc_channel = 'ipc:///share/runtime.ipc'

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
        
        self.samples_manifest = self.read_samples_manifest()
        self.targets_manifest = self.read_targets_manifest()
        self.samples_nfs_paths = self.get_samples_nfs_paths()
        self.targets_nfs_paths = self.get_targets_nfs_paths()
        
        self.num_partitions = 1 if self.lazy else len(self.samples_manifest)
        self._load_partition_data()
        self._miss_queue = multiprocessing.Queue()
        self._handle_miss_proc = multiprocessing.Process(target=self._handle_miss, daemon=True)
        self._handle_miss_proc.start()
        self.cache_hits = 0
    
    def _handle_miss(self):
        # grpc consumes high memory
        manager_uri = "dlcpod-manager:50051"
        channel = grpc.insecure_channel(manager_uri)
        cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
        stub = pb_grpc.ManagerStub(channel)
        while True:
            miss_etag = self._miss_queue.get(block=True)
            stub.handle_datamiss(pb.DataMissRequest(cred=cred, etag=miss_etag))
            
    # manifest files for mapping object path from cloud to local
    def read_samples_manifest(self):
        with open('/share/{}_samples_manifests.json'.format(self.dataset_type), 'r') as f:
            return json.load(f)
    def read_targets_manifest(self):
        p = '/share/{}_targets_manifests.json'.format(self.dataset_type)
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
        else:
            return {}
    def get_samples_nfs_paths(self):
        paths = list(self.samples_manifest.values())
        paths = [path.replace('/runtime', '') for path in paths]
        return paths
    def get_targets_nfs_paths(self):
        paths = list(self.targets_manifest.values())
        paths = [path.replace('/runtime', '') for path in paths]
        return paths
    
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
            while True:
                try:
                    val = self.__getItem__(index)
                    self.cache_hits += 1
                    return val
                except FileNotFoundError as ex:
                    self.cache_hits -= 1  # avoid count cache_hist multiple times
                    nfs_path = self.samples_nfs_paths[index]
                    tmpfs_path = '/runtime{}'.format(nfs_path)
                    parent_dir = '/'.join(tmpfs_path.split("/")[:-1])
                    if not os.path.exists(parent_dir):
                        os.system('mkdir -p {}'.format(parent_dir))
                    # print('cache miss: {}'.format(tmpfs_path))
                    if os.path.exists(nfs_path):
                        shutil.copyfile(nfs_path, tmpfs_path) # NFS --> tmpfs
                        return self.__getitem__(index)
                    else:
                        # print("file miss: {}".format(nfs_path))
                        miss_etag = ex.filename.split('/')[-1]
                        self._miss_queue.put(miss_etag)
                        sub_idx = random.randint(index+1, len(self) - 1)
                        return self.__getitem__(sub_idx)
                except:
                    continue
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
        self.autoscale_workers = autoscale_workers
        self.init_tunner()
        
        context = zmq.Context()
        self.socket_req = context.socket(zmq.REQ)
        self.socket_req.connect(init_channel)

        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.connect(ipc_channel)
    
    def init_tunner(self):
        self.tune_iters = 0
        self.realtime_load_perf = defaultdict(list)
        self.history_load_perf = defaultdict(float)

    def _get_iterator(self) -> '_BaseDataLoaderIter':        
        if self.autoscale_workers and self._iterator is not None:
            self.tune_iters = self._iterator._tune_iters
            self.realtime_load_perf = self._iterator._realtime_load_perf
            self.history_load_perf = self._iterator._history_load_perf
            num_workers = self._iterator._active_workers.value
        else:
            num_workers = self.num_workers
        
        self._iterator = _DLCJobDataLoaderIter(self, self.num_batches, num_workers, self.autoscale_workers, self.realtime_load_perf, 
                                               self.history_load_perf, self.tune_iters, self.socket_req, self.socket_pub)

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
    def __init__(self, loader, num_batches, opt_num_workers, autoscale_workers: bool = True, 
                 realtime_load_perf: defaultdict = None, history_load_perf: np.array = None, tune_iters: int = None, 
                 socket_req: zmq.Socket = None, socket_pub: zmq.Socket = None):
        super(_DLCJobDataLoaderIter, self).__init__(loader)
        
        self._num_batches = num_batches
        self._num_workers = opt_num_workers
        self._autoscale_workers = autoscale_workers
        self._prefetch_factor = loader.prefetch_factor
        self._tune_freq = self._num_workers * self._prefetch_factor
        self.lazy = self._dataset.lazy
        
        # sockets for communication with client
        self._socket_req = socket_req
        self._socket_pub = socket_pub
        
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

        # send batch sampler indexes to client
        # instruct client to init Cache
        if self.lazy:
            t = time.time()
            sampler_iter = iter(self._index_sampler)
            batches = [batch for batch in sampler_iter]
            msg = {
                "paths": batches,
                "active_workers": self._active_workers.value,
                "prefetch_factor": self._prefetch_factor
            }
            self._socket_req.send_multipart([b"init", self._dataset.dataset_type.encode('utf-8'), pickle.dumps(msg)])
            self._socket_req.recv()
            print('init: {}'.format(time.time()-t))
            
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
        
        self._tune_iters = tune_iters
        self._realtime_load_perf = realtime_load_perf
        self._history_load_perf = history_load_perf
        self._reset(loader, first_iter=True)
        
    def _reset(self, loader, first_iter=False):    
        super()._reset(loader, first_iter)
        
        self._last_iter_time = None
        self._req_time = []
        self._fetch_time = []
        
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
        self._history_load_perf = defaultdict(float)
        self._realtime_load_perf = defaultdict(list)  # the latest `num_batches` # of load time
    
    def _tune_worker_num(self):
        mean, median = np.mean, np.median
        num_workers = self._active_workers.value
        
        # buffer `num_batches` load time measurements
        if len(self._realtime_load_perf[num_workers]) == self._tune_freq:
            self._realtime_load_perf[num_workers].clear()

        if self._rcvd_idx == 1 or (self._rcvd_idx % self._tune_freq == 0):
            
            # get rid of inacurrant measurements
            if self._realtime_load_perf[num_workers] is None:
                return
            
            # estimate required workers by comparing req and load time
            if len(self._fetch_time) > 0 and len(self._req_time) > 0:
                est_num_workers = math.ceil(median(self._fetch_time) / median(self._req_time))
            else:
                est_num_workers = np.inf
            
            '''
            update worker weights,
            we omit the first batch to avoid data reloading time spent in the reset function
            '''
            if self._rcvd_idx > 1:
                if num_workers in self._history_load_perf:
                    '''
                    due to the measurement jitter, we use the alpha to balance historical and 
                    the latest performance measurement. 
                    '''
                    alpha = 0.4
                    self._history_load_perf[num_workers] = alpha * self._history_load_perf[num_workers] + (1-alpha) * mean(self._realtime_load_perf[num_workers])
                else:
                    self._history_load_perf[num_workers] = mean(self._realtime_load_perf[num_workers])

            if est_num_workers == np.inf:
                new_num_workers = num_workers
            elif est_num_workers > cpu_count:
                # we assume the load time follows a normal distribution
                loc, scale = math.ceil(3*cpu_count/4), math.ceil(cpu_count/4)
                new_num_workers = min(math.ceil(np.random.normal(loc=loc, scale=scale, size=1)[0]), cpu_count)
                new_num_workers = max(1, new_num_workers)
                
                # if the worker number has been sampled, use worker number with min load time
                if new_num_workers in self._history_load_perf:
                    new_num_workers = sorted(self._history_load_perf.items(), key=lambda item: item[1])[0][0]
            else:
                new_num_workers = est_num_workers
            
            # commit the tunning action
            delta = new_num_workers - num_workers
            for _ in range(abs(delta)):
                if delta > 0:
                    self._spawn_worker(worker_id=self._active_workers.value)
                elif delta < 0:
                    self._pause_worker()
            
            # preload data in the new workers
            if delta > 0:
                pos = self._worker_queue_idx_cycle.get_ptr()
                for _ in range(self._prefetch_factor):
                    self._worker_queue_idx_cycle.set_ptr(pos)
                    for _ in range(delta):
                        self._try_put_index()
                    
            if self._history_load_perf[num_workers] is not None:
                self._tune_iters += 1
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
        if self._last_iter_time is not None:
            if len(self._req_time) == self._tune_freq:
                self._req_time.clear()
            self._req_time.append(time.time() - self._last_iter_time)

        '''
        fetch_time would be np.nan in the first iteration
        '''
        if self.lazy:
            msg = {'send_idx': self._send_idx+1, 
                'rcvd_idx': self._rcvd_idx, 
                'active_workers': self._active_workers.value, 
                'req_time': np.mean(self._req_time)}
            self._socket_pub.send_multipart([b"loadCache", self._dataset.dataset_type.encode('utf-8'), pickle.dumps(msg)])
        
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
                        if self.lazy:
                            self._socket_pub.send_multipart([b"expireChunk", self._dataset.dataset_type.encode('utf-8'), b""])
                        self._shutdown_workers()
                    raise StopIteration

                # Now `self._rcvd_idx` is the batch index we want to fetch
                # Check if the next sample has already been generated
                if len(self._reorder_dict[self._rcvd_idx]) == 2:
                    data, active_workers, fetch_time = self._reorder_dict.pop(self._rcvd_idx)[1]
                    data = self._process_data(data)
                    break
                
                assert not self._shutdown and self._tasks_outstanding > 0
                idx, data, active_workers, fetch_time  = self._get_data()
                self._tasks_outstanding -= 1

                if idx != self._rcvd_idx:
                    # store out-of-order samples
                    self._reorder_dict[idx] += ((data, active_workers, fetch_time),)
                else:
                    del self._reorder_dict[idx]
                    data = self._process_data(data)
                    break
            except StopIteration:
                # epoch is down
                if self._partition_idx == self._dataset.num_partitions-1:
                    self._partition_idx = 0
                    raise StopIteration
                else: 
                    # data in the current part have been consumed    
                    self._partition_idx += 1
                    self._dataset._load_partition_data(self._partition_idx)
                    continue
                
        # ensure the `num_workers` is consensus while reading the batch
        # we skip the first batch because the reset function needs to prefetch data synchronously
        if self._rcvd_idx > 1:
            if active_workers is not None and self._last_iter_time is not None:
                if len(self._realtime_load_perf[active_workers]) == self._tune_freq:
                    self._realtime_load_perf[active_workers].clear()
                self._realtime_load_perf[active_workers].append(time.time() - self._last_iter_time)
                
            if fetch_time is not None:
                if len(self._fetch_time) == self._tune_freq:
                    self._fetch_time.clear()
                self._fetch_time.append(fetch_time)
            
            if self._autoscale_workers:
                self._tune_worker_num()
        
        if self.lazy:
            self._socket_pub.send_multipart([b"releaseCache", self._dataset.dataset_type.encode('utf-8'), str(self._rcvd_idx-1).encode('utf-8')])
        self._last_iter_time = time.time()
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
        self._socket_pub.send_multipart([b'stopIteration', self._dataset.dataset_type.encode('utf-8'), b''])
        
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
