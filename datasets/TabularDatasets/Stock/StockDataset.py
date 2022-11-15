import pandas as pd
from DLCJob import *


def train_test_data(seq, steps):
    X, Y = list(), list()
    for i in range(len(seq)):
        sample = i + steps
        if sample > len(seq)-1:
            break
        x, y = seq[i:sample],seq[sample]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


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
            if sample > len(sample)-1:
                break
            x, y = sample[i:sample], sample[sample]
            X.append(x)
            Y.append(y)            
        return X, Y
    
    def sample_reader(self, path: str = None, raw_bytes: bytes = None):
        df = pd.read_csv(path)
        return df
    
    def target_reader(self, path: str = None, raw_bytes: bytes = None):
        labels = pd.read_csv(path)['Label'].to_list()
        return labels
    
    def __getitem__(self, index):
        x = self.samples.iloc[index]
        y = self.targets[index]
        return {'x_data': x, 'y_target': y}
    
    def __len__(self):
        return len(self.samples)