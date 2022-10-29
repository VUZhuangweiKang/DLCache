from ImageDataset import *
from torchvision import transforms
from PIL import Image

IMG_SIZE = (128,128)


class OpenImagesDataset(ImageDataset):
    def __init__(self, dtype='train'):
        super().__init__(dtype)
    
    def __getitem__(self, index: int):
        return self.try_get_item(index)

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



