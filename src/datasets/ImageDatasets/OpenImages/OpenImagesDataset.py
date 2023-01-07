from ImageDataset import *
from torchvision import transforms
from PIL import Image

IMG_SIZE = (128,128)


class OpenImagesDataset(ImageDataset):
    def __init__(self, dtype='train'):
        super().__init__(dtype)

    def __getItem__(self, index):
        path = self.images[index]
        img = Image.open(path)

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.ToTensor()

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

        label = self.labels[index]
        return img, label



