from PIL import Image
from lib.DLCJob import DLCJobDataset
import numpy as np


class ImageDataset(DLCJobDataset):
    def __init__(self, dataset_type: str = 'train', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        super().__init__(dataset_type)
        
    def _process(self, sample_files, target_files=None):
        cls_names = list(set([path.split('/')[-2] for path in sample_files]))
        cls_idx = {cls_names[i]: i for i in range(len(cls_names))}    
        for image in sample_files:
            self.images.append(self.samples_manifest[image])
            self.labels.append(cls_idx[image.split('/')[-2]])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def _getitem(self, index: int):
        path, target = self.images[index], self.labels[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.images)