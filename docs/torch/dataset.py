import bisect
import warnings
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, randperm
from torch._utils import _accumulate

from ... import Generator, Tensor

__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "random_split",
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class IterableDataset(Dataset[T_co]):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]

        >>> # Mult-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
        [3, 4, 5, 6]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Mult-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    """
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


import os
import json
import queue
import threading
import grpc
import databus.dbus_pb2 as pb
import databus.dbus_pb2_grpc as pb_grpc
import random


def read_secret(arg):
    path = '/secret/{}'.format(arg)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data

class DLCJobDataset(Dataset[T_co]):
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
        self._miss_queue = queue.Queue()
        self._handle_miss_proc = threading.Thread(target=self._handle_miss, daemon=True)
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
                    try:
                        # shutil.copyfile(nfs_path, tmpfs_path) # NFS --> tmpfs
                        os.symlink(nfs_path, tmpfs_path)
                        return self.__getitem__(index)
                    except FileNotFoundError:
                        parent_dir = '/'.join(tmpfs_path.split("/")[:-1])   
                        if not os.path.exists(parent_dir):
                            # os.system('mkdir -p {}'.format(parent_dir))
                            os.makedirs(parent_dir, exist_ok=True)
                        else:
                            miss_etag = ex.filename.split('/')[-1]
                            self._miss_queue.put(miss_etag)
                            sub_idx = random.randint(index+1, len(self) - 1)
                            return self.__getitem__(sub_idx)
                except Exception as ex:
                    print(ex)
        else:
            return self.__getItem__(index)
    
    def __len__(self) -> int:
        raise NotImplementedError

def random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]