import pandas as pd
import torch
import sys
sys.path.insert(0, '../')
from DLCJob import DLCJobDataset


class DDoSDataset(DLCJobDataset):
    def __init__(self, dtype):
        super().__init__(dtype)
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size  
    
    def process(self):
        return list(self.samples.values())[0], None
    
    def sample_reader(self, path: str = None, raw_bytes: bytes = None):
        sample = pd.read_csv(path).to_numpy()
        return torch.tensor(sample, dtype=torch.float32)
    
    def __getitem__(self, index):
        entry = self.samples[index]
        X, Y = entry[:-1], entry[-1:]
        return {'x_data': X, 'y_target': Y}
    
    def __len__(self):
        return len(self.samples)