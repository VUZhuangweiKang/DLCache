import os
import logging
import pickle
import hashlib
import nvidia_smi


def get_logger(name=__name__, level:str ='INFO', file=None):
    levels = {"info": logging.INFO, "error": logging.ERROR, "debug": logging.DEBUG}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    logger = logging.getLogger(name)
    logger.setLevel(levels[level.lower()])

    cl = logging.StreamHandler()
    cl.setLevel(levels[level.lower()])
    cl.setFormatter(formatter)
    logger.addHandler(cl)
    
    if file is not None:
        fl = logging.FileHandler(file)
        fl.setLevel(levels[level.lower()])
        fl.setFormatter(formatter)
        logger.addHandler(fl)
    return logger

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def hashing(data):
    if type(data) is not bytes:
        data = pickle.dumps(data)
    return hashlib.sha256(data).hexdigest()

def MessageToDict(message):
    message_dict = {}
    
    for descriptor in message.DESCRIPTOR.fields:
        key = descriptor.name
        value = getattr(message, descriptor.name)
        
        if descriptor.label == descriptor.LABEL_REPEATED:
            message_list = []
            
            for sub_message in value:
                if descriptor.type == descriptor.TYPE_MESSAGE:
                    message_list.append(MessageToDict(sub_message))
                else:
                    message_list.append(sub_message)
            
            message_dict[key] = message_list
        else:
            if descriptor.type == descriptor.TYPE_MESSAGE:
                message_dict[key] = MessageToDict(value)
            else:
                message_dict[key] = value
    
    return message_dict


def get_cpu_free_mem():
    total, used, free, shared, cache, available = map(int, os.popen('free -t -m').readlines()[1].split()[1:])
    return free

def get_gpu_free_mem():
    try:
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        total = 0
        total_free = 0
        total_used = 0
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total += info.total
            total_free += info.total_free
            total_used += info.total_used
        return total_free
    except:
        return 0

def zcat(fpath):
    import subprocess
    ftype = fpath.split('.')[-1]
    if ftype in ['tar', 'gz', 'bz2']:
        s = subprocess.check_output("zcat {} | wc -c".format(fpath), shell=True)
    elif ftype in ["zip"]:
        s = subprocess.check_output("unzip -l {} | tail -n 1 | awk '{print $1}'".format(fpath), shell=True)
    else:
        s = os.path.getsize(fpath)
    return int(s)