import dask.dataframe as dd
from torch.utils.data import Dataset

        
class DDoSDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = dd.read_csv(data_path, blocksize="1e9") # read 1G data
        
    def get_num_batches(self, batch_size):
        return len(self) // batch_size  

    def __getitem__(self, index):
        entry = self.data.iloc[index].compute().to_numpy()
        return entry[:-1], entry[-1:]