import pandas as pd
import torch
from DLCJob import DLCJobDataset


class DDoSDataset(DLCJobDataset):
    def __init__(self, dtype):
        super().__init__(dtype)
    
    def process(self):
        return list(self.samples.values())[0], list(self.targets.values())[0]
    
    def sample_reader(self, path: str = None, raw_bytes: bytes = None):
        X = pd.read_csv(path).to_numpy()
        return torch.tensor(X)
    
    def target_reader(self, path: str = None, raw_bytes: bytes = None):
        Y = pd.read_csv(path).to_numpy()
        return torch.tensor(Y)
    
    def __getitem__(self, index):
        return {'x_data': self.samples.iloc[index], 'y_target': self.targets[index]}
    
    def __len__(self):
        return len(self.samples)