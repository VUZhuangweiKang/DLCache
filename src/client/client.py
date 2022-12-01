import grpc
import signal
import json
import glob
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
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
            
    def registerJob(self, stub):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            for node in job['nodesequence']:
                p = '/runtime/{}'.format(node)
                if not os.path.exists(p):
                    os.mkdir(p)
            
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