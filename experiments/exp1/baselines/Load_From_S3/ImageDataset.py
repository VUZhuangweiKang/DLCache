from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import io
import os
import boto3


BUCKET_NAME = "vuzhuangwei"

def read_secret(key):
    path = '/secret/{}'.format(key)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data


class ImageDataset(Dataset):
    def __init__(self, manifest_key, transform=None, target_transform=None):
        super().__init__()
        session = boto3.Session(
            aws_access_key_id=read_secret('aws_access_key_id'),
            aws_secret_access_key=read_secret('aws_secret_access_key'),
            region_name=read_secret('region_name')
        )
        self.client = session.client('s3')
        manifest = self.client.get_object(Bucket=BUCKET_NAME, Key=manifest_key)['Body'].read()
        manifest = pd.read_csv(io.BytesIO(manifest))
        self.samples = manifest['sample'].to_numpy()
        
        cls_names = manifest['target'].unique()
        cls_idx = {cls_names[i]: i for i in range(len(cls_names))}
        self.targets = [cls_idx[name] for name in manifest['target'].to_list()]
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int):
        path, target = self.samples[index], self.targets[index]
        for _ in range(5):
            try:
                img = self.client.get_object(Bucket=BUCKET_NAME, Key=path)['Body'].read()
                break
            except:
                print('failed to request: {}'.format(path))
        img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self) -> int:
        return len(self.samples)