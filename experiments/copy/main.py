import os, sys
import shutil
import time
import numpy as np
import subprocess


if __name__ == '__main__':
    src_dir = '/129.59.234.238'
    dst_dir = '/tmp'
    files = os.listdir(src_dir)
    samples = []
    for file in files:
        path = os.path.join(src_dir, file)
        if os.path.isfile(path) and len(samples) < 1000:
            samples.append(path)
    
    os_cp_time = []
    for _ in range(5):
        os.system("rm -r /tmp/*")
        t = time.time()
        # cmd = 'tar cPf - {} | (cd {}; tar xPf -)'.format(files, dst_dir)
        for file in samples:
            cmd = 'cp {} {}'.format(file, dst_dir)
            subprocess.call(cmd, shell=True)
        os_cp_time.append(time.time() - t)
    os_cp_time = np.array(os_cp_time)
    print(os_cp_time)
    
    shutil_cp_time = []
    for _ in range(5):
        os.system("rm -r /tmp/*")
        t = time.time()
        for file in samples:
            shutil.copyfile(file, '{}/{}'.format(dst_dir, file.split('/')[-1]), follow_symlinks=True)
        shutil_cp_time.append(time.time() - t)
    shutil_cp_time = np.array(shutil_cp_time)
    print(shutil_cp_time)