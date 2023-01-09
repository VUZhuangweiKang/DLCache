from PIL import Image
from lib.DLCJob import DLCJobDataset


class ImageDataset(DLCJobDataset):
    def __init__(self, dataset_type: str = 'train', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        super().__init__(dataset_type)
        
    def _process(self, sample_files, target_files=None):
        cls_names = list(set([path.split('/')[-2] for path in sample_files]))
        cls_idx = {cls_names[i]: i for i in range(len(cls_names))}    
        for image in sample_files:
            self.images.append(self.samples_manifest[image])
            self.labels.append(cls_idx[image.split('/')[-2]])

    def __getItem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self) -> int:
        return len(self.images)