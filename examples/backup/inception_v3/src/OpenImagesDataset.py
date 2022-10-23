
from DLCJob import DLCJobDataset
from collections import defaultdict
from torchvision import transforms
import os
import numpy as np
import time
from PIL import Image
import io

IMG_SIZE = (128,128)

class OpenImagesDataset(DLCJobDataset):
    def __init__(self, keys, is_train, train_split = 0.9, random_seed = 42, target_transform = None):
        super().__init__(keys)
        self.data_path = keys
        self.classes = []
        class_idx = 0
 


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
    
    def __getitem__(self, index):
        img, target = self.get_data(index), self.get_target(index)

        img = Image.open(io.BytesIO(img))

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.ToTensor()
        img1 = tr(img)

        width, height = img.size
        if min(width, height)>IMG_SIZE[0] * 1.5:
            tr = transforms.Resize(int(IMG_SIZE[0] * 1.5))
            img = tr(img)

        width, height = img.size
        if min(width, height)<IMG_SIZE[0]:
            tr = transforms.Resize(IMG_SIZE)
            img = tr(img)

        tr = transforms.RandomCrop(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)

        if (img.shape[0] != 3):
            img = img[0:3]

        return img, target
    
    def __len__(self):
        return len(self.data)
    
   



