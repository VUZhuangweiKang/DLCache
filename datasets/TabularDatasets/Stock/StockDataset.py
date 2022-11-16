import pandas as pd
import sys
import torch
from torch import tensor
sys.path.insert(0, "../")
from DLCJob import *


class StockDataset(DLCJobDataset):
    def __init__(self, dtype, steps=10):
        self.steps = steps
        super().__init__(dtype)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size  
    
    def process(self):
        samples = list(self.samples.values())[0]
        X, Y = list(), list()
        for i in range(len(samples)):
            sample = i + self.steps
            if sample > len(samples)-1:
                break
            x, y = samples[i:sample], samples[sample]
            X.append(x)
            Y.append(y)            
        return X, Y
    
    def sample_reader(self, path: str = None, raw_bytes: bytes = None):
        X = pd.read_csv(path, index_col='date')
        X.fillna(method='ffill', inplace=True)
        X = X.to_numpy()
        if(np.isnan(X).any()):
            print('Contains NaN....')
        return tensor(X, dtype=torch.float32)
    
    def target_reader(self, path: str = None, raw_bytes: bytes = None):
        Y = pd.read_csv(path, index_col='date')['Label'].to_numpy()
        return tensor(Y, dtype=torch.float32)
    
    def __getitem__(self, index):
        return {'x_data': self.samples[index], 'y_target': self.targets[index]}
    
    def __len__(self):
        return len(self.samples)