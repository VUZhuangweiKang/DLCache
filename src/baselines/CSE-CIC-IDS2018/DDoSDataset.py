import torch
import math
from torch.utils.data import IterableDataset

        
class DDoSDataset(IterableDataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path                                              
        self.info  = self._get_file_info(file_path)                             
        self.start = self.info['start']                                         
        self.end   = self.info['end']   

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end   = self.end
        else:  # multiple workers
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        sample_iterator = self._sample_generator(iter_start, iter_end)
        return sample_iterator
    
    def __len__(self):
        return self.end - self.start
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def _get_file_info(self, file_path):
        info = {"start": 1, "end": 0}
        with open(file_path, 'r') as fin:
            for _ in enumerate(fin):
                info['end'] += 1
        return info
                                                                                
    def _sample_generator(self, start, end):
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start: continue
                if i >= end: return StopIteration()
                items = line.strip().split(',')
                print(len(items))
                items = [float(item) for item in items]
                yield (torch.FloatTensor(items[:-1]), torch.FloatTensor(items[-1:]))