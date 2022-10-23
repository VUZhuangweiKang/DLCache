from ImageDataset import *
from torchvision import transforms
from PIL import Image
import io

IMG_SIZE = (128,128)


class OpenImagesDataset(ImageDataset):
    def __init__(self, dtype='train/samples'):
        super().__init__(dtype)
        self.classes = []
    
    def __process__(self):
        samples = []
        targets = []
        keys = [self.samples[etag]['Key'] for etag in self.samples]
        cls_keys, classes = self.find_classes(keys)
        for i, class_name in enumerate(classes):
            for key in cls_keys[class_name]:
                samples.append(self.samples[key])
                targets.append(i)
        # return samples, targets
        return torch.tensor(samples), torch.tensor(targets)
    
    def __getitem__(self, index: int):
        img, target = self.try_get_item(index)
        return img, target

    def __sample_reader__(self, path: str = None, raw_bytes: bytes = None):
        img = Image.open(path)

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

        return img

    def __len__(self) -> int:
        return len(self.samples)



