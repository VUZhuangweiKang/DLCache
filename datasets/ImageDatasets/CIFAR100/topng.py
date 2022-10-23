import os
from PIL import Image
import numpy as np


ROOT_PATH = '/tmp/cifar-100-python'
TO_ROOT = './cifar100'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict  =  pickle.load(fo, encoding = 'latin1')
    return dict

meta_dict = unpickle(os.path.join(ROOT_PATH,'meta'))
train_dict = unpickle(os.path.join(ROOT_PATH,'train'))
test_dict = unpickle(os.path.join(ROOT_PATH,'test'))

os.mkdir(TO_ROOT)
os.mkdir(os.path.join(TO_ROOT,'train'))
os.mkdir(os.path.join(TO_ROOT,'val'))

for i in range(100):
    os.mkdir(os.path.join(TO_ROOT,'train',str(i)))
    os.mkdir(os.path.join(TO_ROOT,'val',str(i)))
count = 0

data, label  =  np.array(train_dict['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), train_dict['fine_labels']
for i in range(data.shape[0]):
    count  =  count + 1
    img  =  Image.fromarray(data[i])
    img.save(os.path.join(TO_ROOT,'train', str(label[i]), str(count)+'.png'))

data, label  =  np.array(test_dict['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), test_dict['fine_labels']
for i in range(data.shape[0]):
    count  =  count + 1
    img  =  Image.fromarray(data[i])
    img.save(os.path.join(TO_ROOT,'val', str(label[i]), str(count)+'.png'))