import os
from PIL import Image
import numpy as np
import pandas as pd
import glob


src = '.'
dst = './cifar10'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict  =  pickle.load(fo, encoding = 'latin1')
    return dict

def toimage(batch_dict, train=False):
    data  =  np.array(batch_dict['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    manifest = []
    for i in range(data.shape[0]):
        img  =  Image.fromarray(data[i])
        path = os.path.join(dst, 'train' if train else 'test', 'samples', meta['label_names'][batch_dict['labels'][i]])
        if not os.path.exists(path):
            os.system('mkdir -p {}'.format(path))
        s3_key = os.path.join(path, batch_dict['filenames'][i])
        img.save(s3_key)
        manifest.append([s3_key, meta['label_names'][batch_dict['labels'][i]]])
    return manifest


if __name__ == '__main__':
    try:
        os.mkdir(os.path.join(dst,'train'))
        os.mkdir(os.path.join(dst,'test'))
    except:
        pass

    meta = unpickle(os.path.join(src,'batches.meta'))
    manifest = []

    for f in glob.glob("{}/data_batch_*".format(src)):
        train_dict = unpickle(f)
        manifest.extend(toimage(train_dict, train=True))

    manifest = pd.DataFrame(manifest, columns=['sample', 'target'])
    manifest.to_csv( os.path.join(dst, 'train/manifest.csv'), index=False)
    
    manifest = []
    test_dict = unpickle(os.path.join(src,'test_batch'))
    manifest.extend(toimage(test_dict, train=False))
    manifest = pd.DataFrame(manifest, columns=['sample', 'target'])
    manifest.to_csv( os.path.join(dst, 'test/manifest.csv'), index=False)