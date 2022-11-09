from PIL import Image
from DLCJob import DLCJobDataset


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
            if row['sample'] not in self.samples: continue
            samples.append(self.samples[row['sample']])
            targets.append(cls_idx[row["target"]])
        return samples, targets
    
    def __getitem__(self, index: int):
        return self.try_get_item(index)

    def __sample_reader__(self, path: str = None, raw_bytes: bytes = None):
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.transform(img)
        return img
    
    def __len__(self) -> int:
        return len(self.samples)