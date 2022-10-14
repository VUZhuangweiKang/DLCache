import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from torch.autograd import Variable

from OpenImagesDataset import *
from DLCJob import *




def get_model():
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False #Freezing all the layers and changing only the below layers
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(2048,128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128,6))
    model.aux_logits = False
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

def pred_class(img):
    # transform images
    img_tens = transform_tests(img)
    # change image format (3,300,300) to (1,3,300,300) by help of unsqueeze function
    # image needs to be in cuda before predition
    img_im = img_tens.unsqueeze(0).cuda() 
    uinput = Variable(img_im)
    uinput = uinput.to(device)
    out = model(uinput)
    # convert image to numpy format in cpu and snatching max prediction score class index
    index = out.data.cpu().numpy().argmax()    
    return index

def train_batch(x, y, model, opt, loss_fn):
    output = model(x)
#     print(f"type of output - {type(output)}")
    batch_loss = loss_fn(output, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_data_dir = "/kaggle/input/intel-image-classification/seg_train/seg_train"
# test_data_dir = "/kaggle/input/intel-image-classification/seg_test/seg_test/"
# pred_data_dir = "/kaggle/input/intel-image-classification/seg_pred/seg_pred"

# for i in os.listdir(train_data_dir):
#     new_loc = os.path.join(train_data_dir,i)
#     new = new_loc + '/*.jpg'
#     images = glob(new)
#     print(f'{i}:',len(images))

# for i in os.listdir(test_data_dir):
#     new_loc = os.path.join(test_data_dir,i)
#     new = new_loc + '/*.jpg'
#     images = glob(new)
#     print(f'{i}:',len(images))

# classes = os.listdir(train_data_dir)
# classes = {k: v for k,v in enumerate(sorted(classes))}
# print(classes)





# Performing the Image Transformation and Data Augmentation on the 
# train dataset and transformation on Validation Dataset
# convert data to a normalized torch.FloatTensor

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.3,0.4,0.4,0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.425,0.415,0.405),(0.205,0.205,0.205))
])

# Augmentation on test images not needed
transform_tests = torchvision.transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


t = time.time()
random_seed = int(time.time())
# TODO: number of dataset
dataset_train = OpenImagesDataset(keys=['Imagenet-Mini-Obj/val'], target_transform = transform, is_train = True, random_seed=random_seed, num_classes = 10)
dataset_test = OpenImagesDataset(keys=['Imagenet-Mini-Obj/val'], target_transform = transform_tests, is_train = False, random_seed=random_seed, num_classes = 10)

BATCH_SIZE = 12

print('dataset init time: ', time.time()-t)


t = time.time()
data_loader_train = DLCJobDataLoader(dataset_train, BATCH_SIZE, drop_last=True,shuffle=True,num_workers=2)
data_loader_test = DLCJobDataLoader(dataset_test, BATCH_SIZE, drop_last=True,shuffle=True,num_workers=2)
print('dataloader init time: ', time.time()-t)


# model = models.inception_v3(pretrained=True)
# model.parameters

model, loss_fn, optimizer = get_model()

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(10):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(data_loader_train)):
#         print(f"ix - {ix}, {batch}")
        x, y = batch
#         print(f"type of x - {type(x)}, type of y - {type(y)}")
        x, y= x.to(device), y.to(device)
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
        train_epoch_losses.append(batch_loss)        
    train_epoch_loss = np.array(train_epoch_losses).mean()
    train_epoch_accuracy = np.mean(train_epoch_accuracies)        
    print('Epoch:',epoch,'Train Loss:',train_epoch_loss,'Train Accuracy:',train_epoch_accuracy)

    for ix, batch in enumerate(iter(data_loader_test)):
        x, y = batch
        x, y= x.to(device), y.to(device)
        val_is_correct = accuracy(x, y, model)
        validation_loss = val_loss(x, y, model)    
        val_epoch_accuracy = np.mean(val_is_correct)

    print('Epoch:',epoch,'Validation Loss:',validation_loss,'Validation Accuracy:',val_epoch_accuracy)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)



# pred_files = [os.path.join(pred_data_dir, f) for f in os.listdir(pred_data_dir)]
# pred_files[:10]


# model.eval()

# plt.figure(figsize=(20,20))
# for i, images in enumerate(pred_files):
#     # just want 25 images to print
#     if i > 24:break
#     img = Image.open(images)
#     index = pred_class(img)
#     plt.subplot(5,5,i+1)
#     plt.title(classes[index])
#     plt.axis('off')
#     plt.imshow(img)
