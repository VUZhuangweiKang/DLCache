from PIL import Image
from lib.DLCJob import DLCJobDataset
import os


class ImageDataset(DLCJobDataset):
    def __init__(self, dataset_type: str = 'train', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        super().__init__(dataset_type)
        
    def _process(self, samples_manifest: dict, targets_manifest:dict = None):
        cls_idx = {}
        for image in samples_manifest:
            self.images.append(samples_manifest[image])
            label = image.split('/')[-2]
            if label not in cls_idx:
                cls_idx[label] = len(cls_idx)
            self.labels.append(cls_idx[label])

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