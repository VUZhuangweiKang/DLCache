from collections import defaultdict
import pickle
from typing import Callable, Optional
from DLCJob import DLCJobDataset
import torch

class cifar100Dataset(DLCJobDataset):
    def __init__(self, keys, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(keys)
        self.transform = transform
        self.target_transform = target_transform
        self.count = 0
        self.dur = []
    
    def find_classes(self, keys):
        cls_keys = defaultdict(list)
        for x in keys:
            cls_name = x.split('/')[2]
            cls_keys[cls_name].append(x)
        if len(cls_keys) == 0:
            raise FileNotFoundError(f"Couldn't find any class.")
        classes = list(cls_keys.keys())
        return cls_keys, classes

    def __convert__(self):
        samples = []
        targets = []
        cls_keys, classes = self.find_classes(self.keys)
        for i, class_name in enumerate(classes):
            for key in cls_keys[class_name]:
                samples.append(self.data[key])
                targets.append(i)
        return torch.tensor(samples), torch.tensor(targets)
    
    def __getitem__(self, index: int):
        return self.data[index], self.target[index]

    def __len__(self) -> int:
        return len(self.data.size(0))