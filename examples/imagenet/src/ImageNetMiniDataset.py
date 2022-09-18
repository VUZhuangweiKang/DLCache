from collections import defaultdict
import pickle
from typing import Callable, Optional
from DLCJob import DLCJobDataset


class ImageNetDataset(DLCJobDataset):
    def __init__(self, keys, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 shuffle=False):
        super().__init__(keys, shuffle=shuffle)
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
        return samples, targets
    
    def __getitem__(self, index: int):
        img, target = self.get_data(index), self.get_target(index)
        img = pickle.loads(img)
        return img, target

    def __len__(self) -> int:
        return len(self.data)