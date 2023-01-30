import pandas as pd
from lib.DLCJob import DLCJobDataset


class DDoSDataset(DLCJobDataset):
    def __init__(self, dtype='train'):
        super().__init__(dtype)
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size  

    def _process(self, samples_manifest: dict, targets_manifest:dict = None):
        for file in samples_manifest:
            local_path = samples_manifest[file]
            data = pd.read_csv(local_path).to_numpy()
        self._samples = data[:, :-1]
        self._targets = data[:, -1:]
    
    def _load_sample(self, sample_item):
        return sample_item
    
    def _load_target(self, target_item=None):
        return target_item