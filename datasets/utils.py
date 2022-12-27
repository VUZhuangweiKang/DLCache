import os
import psutil
import shutil
import warnings

def read_secret(arg):
    path = '/secret/{}'.format(arg)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data

def clear():
    for svr in os.listdir('/runtime/'):
        p = '/runtime/{}'.format(svr)
        if os.path.exists(p) and len(os.listdir(p)) > 0:
            os.system('rm -r {}/*'.format(p))

def show_memory_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_info()
    memory = info.vms / 1024./ 1024
    print('process {} use vms: {} MB'.format(pid, memory))
