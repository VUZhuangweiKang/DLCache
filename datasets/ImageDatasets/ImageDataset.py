from PIL import Image
from DLCJob import DLCJobDataset
import numpy as np

class ImageDataset(DLCJobDataset):
    def __init__(self, dtype: str = 'train', transform=None):
        super().__init__(dtype)
        self.count = 0
        self.dur = []
        self.transform = transform
    
    def process(self):
        cls_names = self.manifest['target'].unique()
        cls_idx = {cls_names[i]: i for i in range(len(cls_names))}
        
        samples = []
        targets = []

        for _, row in self.manifest.iterrows():
            if row['sample'] not in self.samples or row['target'] not in cls_idx: 
                continue
            samples.append(self.samples[row['sample']])
            targets.append(cls_idx[row["target"]])
        
        np.save('{}_samples.npy'.format(self.dtype), samples)
        np.save('{}_targets.npy'.format(self.dtype), targets)
        return samples, targets

    def sample_reader(self, sample_item):
        img_path = sample_item
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.transform(img)
        return img
    
    def target_reader(self, target_item):
        return target_item