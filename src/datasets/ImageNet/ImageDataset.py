from PIL import Image
from lib.DLCJob import DLCJobDataset


class ImageDataset(DLCJobDataset):
    def __init__(self, dataset_type: str = 'train', transform=None, 
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(dataset_type)
        
    def _process(self, samples_manifest: dict, 
                 targets_manifest:dict = None):
        cls_idx = {}
        for image in samples_manifest:
            self._samples.append(samples_manifest[image])
            label = image.split('/')[-2]
            if label not in cls_idx:
                cls_idx[label] = len(cls_idx)
            self._targets.append(cls_idx[label])
    
    def _load_sample(self, sample_item):
        with Image.open(sample_item) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_target(self, target_item=None):
        if self.target_transform is not None:
            target_item = self.target_transform(target_item)
        return target_item