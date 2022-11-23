import os
import glob
import pandas as pd


nfs_nodes = ['129.59.234.236', "129.59.234.237", "129.59.234.238", 
             "129.59.234.239", "129.59.234.240", "129.59.234.241"]

meta_file = '/129.59.234.240/cifar-10-batches-py/batches.meta'
train_imgs = glob.glob('/129.59.234.240/cifar-10-batches-py/train/samples/*/*.png')
test_imgs = glob.glob('/129.59.234.240/cifar-10-batches-py/test/samples/*/*.png')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict  =  pickle.load(fo, encoding = 'latin1')
    return dict


if __name__ == '__main__':
    meta = unpickle(meta_file)
    cls_names = meta['label_names']
    cls_dict = {cls_names[i]: i for i in range(len(cls_names))}
    
    targets = []
    paths = []
    for i, img in enumerate(train_imgs):
        img = img.split("/")
        img[1] = nfs_nodes[i%6]
        paths.append('/'.join(img))
        name = img[5]
        targets.append(cls_dict[name])
    train_manifest = pd.DataFrame({'sample': paths, 'target': targets})
    train_manifest.to_csv('train_manifest.csv', index=False)
    
    targets = []
    paths = []
    for i, img in enumerate(test_imgs):
        img = img.split("/")
        img[1] = nfs_nodes[i%6]
        paths.append('/'.join(img))
        name = img[5]
        targets.append(cls_dict[name])
    test_manifest = pd.DataFrame({'sample': paths, 'target': targets})
    test_manifest.to_csv("test_manifest.csv", index=False)
    print(test_manifest)
        