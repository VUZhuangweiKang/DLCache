import numpy as np
import pandas as pd

if __name__ == '__main__':
    train_samples = np.load('train_samples.npy')
    train_targets = np.load("train_targets.npy")
    train_manifest = pd.DataFrame(zip(train_samples, train_targets), columns=["sample", "target"])
    train_manifest.to_csv("train_manifest.csv", index=False)

    val_samples = np.load('train_samples.npy')
    val_targets = np.load("train_targets.npy")
    val_manifest = pd.DataFrame(zip(val_samples, val_targets), columns=["sample", "target"])
    val_manifest.to_csv("val_manifest.csv", index=False)
    
    