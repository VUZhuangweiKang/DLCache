import numpy as np


if __name__ == '__main__':
    tar_perf = np.load("nfs_cp.npy")
    shutil_perf = np.load("shutil_cp.npy")
    print(tar_perf)
    print('------------')
    print(shutil_perf)
    