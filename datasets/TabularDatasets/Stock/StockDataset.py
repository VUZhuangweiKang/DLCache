import pandas as pd
import sys
import torch
from torch import tensor
sys.path.insert(0, "../")
from DLCJob import *


class StockDataset(DLCJobDataset):
    def __init__(self, dtype, steps=10):
        self.steps = steps
        self.X = []
        self.Y = []
        super().__init__(dtype)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size  
    
    def _process(self, sample_files, target_files=None):
        data = pd.read_csv(sample_files[0], index_col='date')
        data.fillna(method='ffill', inplace=True)
        data = data.to_numpy()
        if(np.isnan(data).any()):
            print('Contains NaN....')
        data = tensor(data, dtype=torch.float32)
    
        targets = pd.read_csv(target_files[0], index_col='date')['Label'].to_numpy()
        targets = tensor(targets, dtype=torch.float32)
    
        for i in range(len(data)):
            sample = i + self.steps
            if sample > len(data)-1:
                break
            x, y = data[i:sample], data[sample]
            self.X.append(x)
            self.Y.append(y)            
    
    def __getitem__(self, index):
        return {'x_data': self.X[index], 'y_target': self.Y[index]}
    
    def __len__(self):
        return len(self.X)