import grpc
import signal
import json
import glob
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import time
import shutil
import concurrent.futures
import multipledispatch
from utils import *


logger = get_logger(__name__, level='Debug')
cloudSecret = {
    "aws_access_key_id": read_secret('aws_access_key_id'),
    "aws_secret_access_key": read_secret('aws_secret_access_key'),
    "region_name": read_secret('region_name')   
}
manager_uri = "dlcpod-manager:50051"


class Client(object):
    def __init__(self):
        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        channel = grpc.insecure_channel(manager_uri)
        # register the job to Manager
        if len(self.jobsmeta) > 0:
            self.cred = pb.Credential(username=read_secret('dlcache_user'), password=read_secret('dlcache_pwd'))
            
            conn_stub = pb_grpc.ConnectionStub(channel)
            req = pb.ConnectRequest(
                cred=self.cred, 
                s3auth=pb.S3Auth(**cloudSecret),
                createUser=True
            )
            resp = conn_stub.connect(req)
            if resp.rc == pb.RC.FAILED:
                logger.error("failed to connect to server with: {}".format(resp.resp))
                raise Exception
            else:
                logger.info("connect to server")
            
            stub = pb_grpc.RegistrationStub(channel)
            self.registerJob(stub)
    
    def prefetch(self, idx=None):
        def docopy(nfs_path):
            if os.path.exists(nfs_path):
                tmpfs_path = '/runtime{}'.format(nfs_path)
                root_folder = '/'.join(tmpfs_path.split('/')[:-1])
                if not os.path.exists(root_folder):
                    os.makedirs(root_folder)
                shutil.copyfile(nfs_path, tmpfs_path)  # NFS --> tmpfs
                assert os.stat(nfs_path).st_size == os.stat(tmpfs_path).st_size
                return True, nfs_path
            print('failed to copy {}'.format(nfs_path))
            return False, nfs_path
        
        if idx is None:
            idx = self._send_idx

        if idx < len(self.nfs_paths):
            t = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for sample_path, target_path in self.nfs_paths[idx]:
                    futures.append(executor.submit(docopy, sample_path))
                    if target_path is not None:
                        futures.append(executor.submit(docopy, target_path))

            for future in concurrent.futures.as_completed(futures):
                rc, miss_file = future.result()
                etag = miss_file.split('/')[-1]
                if not rc:
                    self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, etag=etag))
            self._send_idx += 1
            self.load_time.append(time.time()-t)
            
    def registerJob(self, stub):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            
            train_keys = ds['keys']['train']
            val_keys = ds['keys']['validation']
            test_keys = ds['keys']['test']
            request = pb.RegisterRequest(
                cred=self.cred,
                datasource=pb.DataSource(
                    name=ds['name'], 
                    bucket=ds['bucket'], 
                    keys=pb.JobDatasets(
                        train=pb.Dataset(**train_keys),
                        validation=pb.Dataset(**val_keys),
                        test=pb.Dataset(**test_keys)
                    )
                ),
                nodesequence=job['nodesequence'],
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
                job['jobId'] = resp.jobId
                job['mongoUri'] = resp.mongoUri
                with open('/share/{}.json'.format(job['name']), 'w') as f:
                    json.dump(job, f)
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jobId))


if __name__ == '__main__':
    client = Client()