from collections import defaultdict
import pickle
from PIL import Image
from DLCJob import DLCJobDataset


class ImageNetDataset(DLCJobDataset):
    def __init__(self, dtype: str = 'train', transform=None):
        super().__init__(dtype)
        self.count = 0
        self.dur = []
        self.transform = transform
    
    def find_classes(self, keys):
        cls_keys = defaultdict(list)
        for x in keys:
            cls_name = x.split('/')[2]
            cls_keys[cls_name].append(x)
        if len(cls_keys) == 0:
            raise FileNotFoundError(f"Couldn't find any class.")
        classes = list(cls_keys.keys())
        return cls_keys, classes

    def __process__(self):
        samples = []
        targets = []
        keys = [self.samples[etag]['Key'] for etag in self.samples]
        cls_keys, classes = self.find_classes(keys)
        for i, class_name in enumerate(classes):
            for key in cls_keys[class_name]:
                samples.append(self.samples[key])
                targets.append(i)
        return samples, targets
    
    def __getitem__(self, index: int):
        img, target = self.try_get_item(index)
        return img, target

    # def __sample_reader__(self, path: str = None, raw_bytes: bytes = None):
    #     img = Image.open(path)
    #     img = img.convert("RGB")
    #     img = self.transform(img)
    #     return img

    def __sample_reader__(self, path: str = None, raw_bytes: bytes = None):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)