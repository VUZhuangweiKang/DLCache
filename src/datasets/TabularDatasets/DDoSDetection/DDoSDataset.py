import pandas as pd
import torch
import sys
sys.path.insert(0, '../')
from lib.DLCJob import DLCJobDataset


class DDoSDataset(DLCJobDataset):
    def __init__(self, dtype='train'):
        super().__init__(dtype)
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size  
    
    def _process(self, sample_files, target_files=None):
        self.data = torch.tensor(pd.read_csv(sample_files[0]).to_numpy(), dtype=torch.float32)
    
    def __getItem__(self, index):
        entry = self.data[index]
        X, Y = entry[:-1], entry[-1:]
        return {'x_data': X, 'y_target': Y}
    
    def __len__(self):
        return len(self.data)