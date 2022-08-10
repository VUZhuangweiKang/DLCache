import pickle
import time
import sys
from typing import Callable, Optional
from DLCJob import DLCJobDataset


summary = []
class ImageNetDataset(DLCJobDataset):
    def __init__(self, keys, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(keys)
        self.transform = transform
        self.target_transform = target_transform
        self.count = 0
        self.dur = []
    
    def find_classes(self, keys):
        cls_keys = {}
        classes = []
        for x in keys:
            clas = x.split('/')[2]
            classes.append(clas)
            if clas in cls_keys:
                cls_keys[clas].append(x)
            else:
                cls_keys[clas] = [x]
        classes = sorted(classes)
        if len(classes) == 0:
            raise FileNotFoundError(f"Couldn't find any class.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return cls_keys, class_to_idx

    def __convert__(self):
        samples = []
        targets = []
        cls_keys, class_to_idx = self.find_classes(self.keys)
        for target_class in class_to_idx:
            for key in cls_keys[target_class]:
                samples.append(self.data[key])
                targets.append(class_to_idx[target_class])
        return samples, targets
    
    def __getitem__(self, index: int):
        global summary
        t = time.time()
        img, target = self.get_data(index), self.get_target(index)
        img_size = sys.getsizeof(img)
        img = pickle.loads(img)
        e = time.time()
        with open('summary.csv', 'a+') as f:
            f.write(str(e-t)+'\n')
        with open('throughput.csv', 'a+') as f:
            f.write(str(img_size/(e-t))+'\n')
        return img, target

    def __len__(self) -> int:
        return len(self.data)