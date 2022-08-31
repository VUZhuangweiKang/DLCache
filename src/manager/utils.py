import logging
from google.protobuf.timestamp_pb2 import Timestamp
import pickle
import hashlib
import boto3


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

grpc_ts = lambda ts: Timestamp(seconds=int(ts), nanos=int(ts % 1 * 1e9))

def hashing(data):
    if type(data) is not bytes:
        data = pickle.dumps(data)
    return hashlib.sha256(data).hexdigest()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def parse_config(section: str) -> dotdict:
        with open("/config/{}".format(section), 'r') as f:
            config_str = f.readlines()
        result = {}
        for item in map(lambda x: x.split("="), config_str):
            result[item[0]] = item[1]
        return dotdict(result)

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


class S3Helper:
    def __init__(self, s3auth: dict):
        session = boto3.Session(
            aws_access_key_id=s3auth['aws_access_key_id'],
            aws_secret_access_key=s3auth['aws_secret_access_key'],
            region_name=s3auth['region_name']
        )from __future__ import print_function
import grpc
import signal
import json
import time
import glob
from pathlib import Path
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import pyinotify
import shutil
import threading
from collections import OrderedDict
import numpy as np
from pymongo import MongoClient
import datetime
import bson
from utils import *


logger = get_logger(__name__, level='Debug')


class Client(pyinotify.ProcessEvent):
    def __init__(self):
        def read_secret(arg):
            path = '/secret/{}'.format(arg)
            assert os.path.exists(path)
            with open(path, 'r') as f:
                data = f.read().strip()
            return data

        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        self.runtime_buffer = OrderedDict()
        self.req_time = []
        self.load_time = []
        
        # runtime tmpfs waterline: n*num_workers*batch_size, n=2 initially
        self.waterline = 2
        self.pidx = 0
        try:
            with open('/share/prefetchKeys.json', 'r') as f:
                tmp = json.load(f)
                self.runtime_conf = tmp['meta']
                self.prefetchPaths = tmp['policy']
        except FIleNotFoundError:
            self.prefetchPaths = None
            
        # create inotify
        wm = pyinotify.WatchManager()
        mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_NOWRITE
        wm.add_watch("/share", mask, auto_add=True, rec=True)
        wm.add_watch('/runtime', mask, auto_add=True, rec=True)
        
        # watch data access events
        nfs_servers = os.popen(cmd="df -h | grep nfs | awk '{ print $6 }'").read().strip().split('\n')
        for svr in nfs_servers:
            wm.add_watch(svr, mask, auto_add=True, rec=True)

        self.notifier = pyinotify.Notifier(wm, self)
        notifierThread = threading.Thread(target=self.notifier.loop)
        
        if len(self.jobsmeta) > 0:
            self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
            self.channel = grpc.insecure_channel("dlcpod-manager:50051")
            self.conn_stub = pb_grpc.ConnectionStub(self.channel)
            
            req = pb.ConnectRequest(
                cred=self.cred, 
                s3auth=pb.S3Auth(
                    aws_access_key_id=read_secret('aws_access_key_id'),
                    aws_secret_access_key=read_secret('aws_secret_access_key'),
                    region_name=read_secret('region_name')
                ),
                createUser=True
            )
            resp = self.conn_stub.connect(req)
            if resp.rc == pb.RC.FAILED:
                logger.error("failed to connect to server with: {}".format(resp.resp))
                raise Exception
            else:
                logger.info("connect to server")
            
            self.register_stub = pb_grpc.RegistrationStub(self.channel)
            self.register_job()
            self.datamiss_stub = pb_grpc.DataMissStub(self.channel)
                        
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
        
        self.dataset_col = None
            
    def exit_gracefully(self):
        self.channel.close()
        self.notifier.stop()
            
    def register_job(self):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['dataSource']
            if 'keys' not in ds: ds['keys'] = []
            request = pb.RegisterRequest(
                cred=self.cred,
                datasource=pb.DataSource(name=ds['name'], bucket=ds['bucket'], keys=ds['keys']),
                nodeSequence=job['nodeSequence'],
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = self.register_stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))
            with open('/share/{}.json'.format(job['name']), 'w') as f:  # marshelled registration response
                json.dump(MessageToDict(resp), f)
            
            mongo_client = MongoClient(resp.mongoUri)
            self.dataset_col = mongo_client.Cacher.Datasets

            # prefetch the first 2 batches
            # while not self.prefetchPaths: pass
            for _ in range(self.waterline): 
                self.prefetch()

    def handle_datamiss(self):
        with open('/share/datamiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            bucket, key = key.split(':')
            resp = self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, bucket=bucket, key=key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))

    def prefetch(self):
        if self.runtime_conf['LazyLoading']:
            for _ in range(self.runtime_conf['num_workers']):
                for path in self.prefetchPaths[self.pidx]:
                    if self.pidx not in self.runtime_buffer:
                        self.runtime_buffer[self.pidx] = []
                    if path not in self.runtime_buffer[self.pidx]:
                        copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
                        self.runtime_buffer[self.pidx].append(path)
                self.pidx += 1
        else:
            path = self.prefetchPaths[self.pidx]
            copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
            self.pidx += 1

    # debug: next有写入，但是client端没有反应
    def process_IN_CREATE(self, event):
        if event.pathname == '/share/datamiss':
            return self.handle_datamiss()
        elif event.pathname == '/share/next':
            self.req_time.append(time.time())
            self.prefetch()
        elif event.path == '/share/prefetchKeys.json':
            with open('/share/prefetchKeys.json', 'r') as f:
                tmp = json.load(f)
                self.runtime_conf = tmp['meta']
                self.prefetchPaths = tmp['policy']

    def process_IN_MODIFY(self, event):
        self.process_IN_CREATE(event)
            
    def process_IN_CLOSE_NOWRITE(self, event):
        path = event.pathname
        if '/runtime' in path:
            # pop head
            batch_index = self.runtime_buffer.keys[0]
            self.runtime_buffer[batch_index].pop(0)
            if len(self.runtime_buffer[batch_index]) == 0:
                self.runtime_buffer.popitem(last=False)
            shutil.rmtree(path, ignore_errors=True)
            
            # tune buffer size
            if len(self.req_time) > 1:
                # decide alpha and beta based on the latest 3 measurements
                alpha = np.diff(self.req_time[-4:])[1:]
                """
                To ensure the data is always available for DataLoader, the length of buffer should be:
                s >= 2*B, if alpha >= beta; otherwise,
                s >= (N-k)*(1-alpha/beta) 
                """
                s = max(2, np.mean((1-alpha/beta)*(N-k), dtype=int))
                
                # update waterline according to load/consume speed
                if self.waterline == s:
                    return
                else:
                    self.waterline = s
                    while len(self.runtime_buffer) > s:
                        self.runtime_buffer.popitem(last=False)
                        self.pidx -= 1
                    while len(self.runtime_buffer) < s:
                        self.prefetch()
        elif path.split('/')[0] in nfs_servers:
            assert self.dataset_col is not None
            etag = path.split('/')[1]
            now = datetime.utcnow().timestamp()
            self.dataset_col.update_one(
                {"ETag": etag}, 
                {
                    "$set": {"LastAccessTime": bson.timestamp.Timestamp(int(now), inc=1)},
                    "$inc": {"TotalAccessTime": 1}
                })


if __name__ == '__main__':
    if not os.path.exists("/share/datamiss"):
        Path("/share/datamiss").touch()
    Client()
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        results = []
        for page in pages:
            for info in page['Contents']:
                results.append(info)
        return results
    
    def get_object(self, bucket_name, key):
        return self.client.get_object(Bucket=bucket_name, Key=key)['Body'].read()


def csv_has_header(path):
    import csv
    with open(path, 'rb') as csvfile:
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(csvfile.read(2048))
        csvfile.seek(0)
        return has_header
    
def copyfile(src, dst):
    assert os.path.exists(src)
    base_dir = '/'.join(dst.split('/')[:-1])
    if not os.path.exists(base_dir):
        os.system('mkdir -p {}'.format(base_dir))
    shutil.copy(src, dst)
    