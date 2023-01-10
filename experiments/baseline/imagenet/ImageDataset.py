from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, manifest_path, transform=None, target_transform=None):
        super().__init__()
        manifest = pd.read_csv(manifest_path)
        self.samples = manifest['sample'].to_numpy()
        
        cls_names = manifest['target'].unique()
        cls_idx = {cls_names[i]: i for i in range(len(cls_names))}
        self.targets = [cls_idx[name] for name in manifest['target'].to_list()]
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int):
        path, target = self.samples[index], self.targets[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self) -> int:
        return len(self.samples)