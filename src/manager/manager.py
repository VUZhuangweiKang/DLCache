import os
import sys
import shutil
import grpc
import boto3, botocore
import json, bson
import configparser
import multiprocessing
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
import concurrent.futures
from collections import defaultdict
from pymongo.mongo_client import MongoClient
from utils import *


logger = get_logger(name=__name__, level='debug')


# def download_file(client, bucket, key, path):
#     if os.path.exists(path):
#         return
#     import tarfile, zipfile
#     tmp_file = '/tmp/{}'.format(path.split('/')[-1])
#     logger.info("downloading file {} ...".format(key))
#     try:
#         client.download_file(bucket, key, tmp_file)
#     except botocore.exceptions.ClientError as e:
#         if e.response["Error"]["Code"] == "404":
#             logger.error("Object {} does not exist".format(key))
#         return False
    
#     if tarfile.is_tarfile(tmp_file):
#         read_type = "r:gz" if tmp_file.endswith("tgz") else "r"
#         with tarfile.open(tmp_file, read_type) as mytar:
#             mytar.extractall(path, members=mytar)
#         os.remove(tmp_file)
#     elif zipfile.is_zipfile(tmp_file):
#         with zipfile.ZipFile(tmp_file) as myzip:
#             myzip.extractall(path)
#         os.remove(tmp_file)
#     else:
#         shutil.move(tmp_file, path)
#     return True    


def download_file(client, bucket, key, path):
    if os.path.exists(path):
        return
    tmp_file = '/tmp/{}'.format(path.split('/')[-1])
    logger.info("downloading file {} ...".format(key))
    try:
        client.download_file(bucket, key, tmp_file)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.error("Object {} does not exist".format(key))
        return False
    
    if tmp_file.endswith("tar.gz") or tmp_file.endswith('tgz'):
        os.system("tar --use-compress-program=pigz -xvpf {} -C {}".format(tmp_file, path))
        os.remove(tmp_file)
    elif tmp_file.endswith('zip'):
        os.system("unzip {} -d {}".format(tmp_file, path))
        os.remove(tmp_file)
    else:
        shutil.move(tmp_file, path)
    return True    
        
        
class Manager():
    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        
        try:
            self.managerconf = parser['manager']
            mconf = parser['mongodb']
        except KeyError as err:
            logger.error(err)
        
        self.mongoUri = "mongodb://{}:{}@{}:{}".format(mconf['username'], mconf['password'], mconf['host'], mconf['port'])
        mongo_client = MongoClient(self.mongoUri)        
        
        def load_collection(name, schema):
            collections = mongo_client.Cacher.list_collection_names()
            with open(schema, 'r') as f:
                schema = json.load(f)
            if name not in collections:
                return mongo_client.Cacher.create_collection(name=name, validator={"$jsonSchema": schema}, validationAction="error")
            else:
                return mongo_client.Cacher[name]
        self.client_col = load_collection('Client', "mongo-schemas/client.json")
        self.job_col = load_collection('Job', "mongo-schemas/job.json")
        self.dataset_col = load_collection('Datasets', "mongo-schemas/datasets.json")
        logger.info("start global manager")

    def auth_client(self, cred, s3auth=None, conn_check=False):
        result = self.client_col.find_one(filter={"Username": cred.username})
        if result is None:
                rc = pb.RC.NO_USER
        else:
            if cred.password == result['Password']:
                if conn_check:
                    rc = pb.RC.CONNECTED if result['Status'] else pb.RC.DISCONNECTED
                else:
                    rc = pb.RC.CONNECTED
            else:
                rc = pb.RC.WRONG_PASSWORD
        
        # check whether to update s3auth information
        if rc == pb.RC.CONNECTED and s3auth is not None and result['S3Auth'] != s3auth:
            result = self.client_col.update_one(
                filter={"Username": cred.username}, 
                update={"$set": {"S3Auth": s3auth}})
            if result.modified_count != 1:
                logger.error("user {} is connected, but failed to update S3 authorization information.".format(cred.username))
                rc = pb.RC.FAILED
        return rc
    
    def get_s3_client(self, cred):
        result = self.client_col.find_one(filter={"$and": [{"Username": cred.username, "Password": cred.password}]})
        s3auth = result['S3Auth']
        s3_session = boto3.Session(
            aws_access_key_id=s3auth['aws_access_key_id'],
            aws_secret_access_key=s3auth['aws_secret_access_key'],
            region_name=s3auth['region_name']
        )
        s3_client = s3_session.client('s3')
        return s3_client

    def filter_chunks(self, page: list):
        etags = {}
        for i in range(len(page)):
            if page[i]['Size'] == 0: continue
            page[i]["ETag"] = page[i]["ETag"].strip('"')  # ETag value from S3 contains " sometimes
            etags[page[i]["ETag"]] = page[i]["LastModified"]
            
        results = self.dataset_col.aggregate(pipeline=[{"$match": {"ETag": {"$in": list(etags.keys())}}}])
        existing_etags = {item['ETag']: item for item in results}

        chunks = []
        for info in page:
            if info['Size'] == 0: continue
            lastModified = bson.timestamp.Timestamp(int(info['LastModified'].timestamp()), inc=1)
            if info['ETag'] not in existing_etags or lastModified > existing_etags[info['ETag']]['LastModified']:
                info['Exist'] = False
            else:
                info = existing_etags[info['ETag']]
                info['Exist'] = True
            info['ChunkETag'] = info['ETag']
            chunks.append(info)
        return chunks
    
    # Hybrid Data Eviction: evict dataobj from a specific NFS server
    # 1. data objs without any binding running jobs are first removed based on the LFU policy.
    # 2. regarding the datasets being used, we adopt the LRU policy
    def data_eviction(self, node=None, require=None):
        def helper(pipeline):
            nonlocal require
            if node:
                pipeline.append({"$match": {"Location": node}})
            rmobjs = self.dataset_col.aggregate(pipeline)
            for obj in rmobjs:
                if require <= 0: break
                path = '/{}/{}'.format(obj['Location'], obj['ChunkETag'])
                if os.path.exists(path):
                    if os.path.isdir(path):
                        os.rmdir(path)
                    else:
                        os.remove(path)
                    require -= obj['ChunkSize']
            self.dataset_col.delete_many({"ChunkETag": [obj['ChunkETag'] for obj in rmobjs]})
            
        lfu_pipeline = [{"$project": {'Location': 1, 'ChunkETag': 1, 'TotalAccessTime': 1, 'ChunkSize': 1, 'Jobs': 1, 'num_jobs': {'$size': '$Jobs'}}},
                        {"$match": {"num_jobs": {"$eq": 0}}},
                        {"$sort": {"TotalAccessTime": 1}}]
        helper(lfu_pipeline)
        
        if require > 0:
            lru_pipline = [{"$project": {'Location': 1, 'ChunkETag': 1, 'LastAccessTime': 1, 'ChunkSize': 1, 'Jobs': 1, 'num_jobs': {'$size': '$Jobs'}}}, 
                           {"$match": {"num_jobs": {"$gt": 0}}},
                           {"$sort": {"LastAccessTime": 1}}]
            helper(lru_pipline)

    # Dataset Rebalance: if some data objects from a dataset is already existing
    def move_chunk(self, chunk, node_sequence):
        while True:
            for node in node_sequence:
                # if shutil.disk_usage("/{}".format(node))[-1] >= dataobj['ChunkSize']:
                try:
                    if node != chunk['Location']:
                        src_path = '/{}/{}'.format(chunk['Location'], chunk['ChunkETag'])
                        dst_path = '/{}/{}'.format(node, chunk['ChunkETag'])
                        if not os.path.exists(dst_path):
                            os.rename(src=src_path, dst=dst_path)
                        self.dataset_col.update_one({"ChunkETag": chunk['ChunkETag']}, {"$set": {"Location": node}})
                        chunk['Location'] = node
                    return [chunk]
                except OSError:  # handle the case that the node is out of space
                    continue
            self.data_eviction(node=node_sequence[0], require=chunk['ChunkSize'])
    
    def clone_chunk(self, chunk: dict, s3_client, bucket_name, chunk_size, node_sequence, part=None, miss=False):
        key = chunk['Key']
        chunk["Bucket"] = bucket_name
        if not miss:
            chunk['LastModified'] = bson.timestamp.Timestamp(int(chunk['LastModified'].timestamp()), inc=1)
            chunk['TotalAccessTime'] = 0
        now = datetime.utcnow().timestamp()
        chunk['LastAccessTime'] = bson.timestamp.Timestamp(int(now), inc=1)
        file_type = key.split('.')[-1].lower()

        def assignLocation(obj):
            # try to assign the object follow the node sequence
            # if failed, evict data
            while True:
                for node in node_sequence:
                    free_space = shutil.disk_usage("/{}".format(node))[-1]
                    if free_space >= obj['ChunkSize']:
                        obj['Location'] = node
                        return obj
                self.data_eviction(node=node_sequence[0], require=obj['ChunkSize'])

        chunk_size *= 1e6 # Mill bytes --> bytes
        if chunk['Size'] <= chunk_size:
            if 'ETag' in chunk:
                etag = chunk['ETag'].strip("\"")
            else:
                value = s3_client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
                etag = hashing(value)
            chunk["Part"] = 0
            chunk['ChunkETag'] = etag
            chunk['ChunkSize'] = chunk['Size']
            if not miss:
                chunk = assignLocation(chunk)
            path = "/{}/{}".format(chunk['Location'], etag)
            download_file(s3_client, bucket_name, key, path)
            obj_chunks = [chunk]
        else:
            # large file
            obj_chunks = []
            path = '/tmp/{}'.format(key)
            download_file(s3_client, bucket_name, key, path)
            if file_type in ['csv', 'parquet']:
                from dask import dataframe as DF
                if file_type == 'csv':
                    chunks = DF.read_csv(path, index=False)
                elif file_type == 'parquet':
                    chunks = DF.read_parquet(path)

                chunks = chunks.repartition(partition_size=chunk_size)
                for i, chunk_part in enumerate(chunks.partitions):
                    if part is None or part == i:
                        etag = hashing(chunk_part.compute())
                        chunk['ChunkSize'] = chunk_part.memory_usage_per_partition(deep=True).compute()
                        chunk = assignLocation(chunk)
                        path = "/%s/%s" % (chunk['Location'], etag)
                        if file_type == 'csv':
                            chunk_part.to_csv(path, index=False)
                        elif file_type == 'parquet':
                            chunk_part.to_parquet(path='/', write_metadata_file=False, name_function=lambda _: path)
                        chunk['ChunkETag'] = etag
                        chunk['Part'] = i
                        obj_chunks.append(chunk)
            elif file_type == 'npy':
                import numpy as np
                with open(path, 'rb') as f:
                    major, minor = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                    assert not fortran, "Fortran order arrays not supported"
                    # Get the number of elements in one 'row' by taking a product over all other dimensions.
                    row_size = np.prod(shape[1:])
                    start_row = 0
                    num_rows = chunk_size/(row_size*dtype.itemsize)
                    p = 0
                    while start_row < shape[0]:
                        if part is None or part == p:
                            start_byte = start_row * row_size * dtype.itemsize
                            f.seek(start_byte, 1)
                            if start_row+num_rows > shape[0]:
                                num_rows = shape[0]-start_row+1
                            n_items = row_size * num_rows
                            value = np.fromfile(f, count=n_items, dtype=dtype)
                            value = value.reshape((-1,) + shape[1:])
                            etag = hashing(value)
                            chunk['ChunkSize'] = value.nbytes
                            chunk = assignLocation(chunk)
                            path = "/{}/{}".format(chunk['Location'], etag)
                            np.save(path, value)
                            chunk['ChunkETag'] = etag
                            chunk['Part'] = p
                            obj_chunks.append(chunk)
                        start_row += num_rows
                        p += 1
            else:
                with open(path, 'rb') as f:
                    f.seek(0)
                    value = f.read(chunk_size)
                    p = 0
                    while value:
                        if part is None or part == p:
                            etag = hashing(value)
                            chunk['ChunkSize'] = sys.getsizeof(value)
                            chunk = assignLocation(chunk)
                            path = "/{}/{}".format(chunk['Location'], etag)
                            with open(path, 'wb') as f:
                                f.write(value)
                            chunk['ChunkETag'] = etag
                            chunk['Part'] = p
                            obj_chunks.append(chunk)
                        value = f.read(chunk_size)
                        p += 1
        return obj_chunks


class ConnectionService(pb_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        cred, s3auth = request.cred, request.s3auth
        rc = self.manager.auth_client(cred=cred, s3auth=MessageToDict(s3auth))
        if rc == pb.RC.WRONG_PASSWORD:
            resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="wrong password")
        elif rc == pb.RC.NO_USER:
            if request.createUser:
                try:
                    result = self.manager.client_col.insert_one({
                        "Username": cred.username,
                        "Password": cred.password,
                        "S3Auth": MessageToDict(s3auth),
                        "Status": True
                    })
                except Exception as ex:
                    print(ex)
                if result.acknowledged:
                    logger.info("user {} connected".format(cred.username))
                    resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                else:
                    resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp = "not found user {}".format(cred.username))
        elif rc == pb.RC.DISCONNECTED:
            result = self.manager.client_col.update_one(
                filter={
                    "Username": cred.username,
                    "Password": cred.password,
                },
                update={"$set": {"Status": True, "Jobs": []}}
            )
            if result['modified_count'] == 0:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                logger.info("user {} connected".format(cred.username))
        else:
            resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
            logger.info("user {} connected".format(cred.username))
        return resp


class RegistrationService(pb_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
    
    def register(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        jobId = "{}-{}".format(cred.username, request.datasource.name)
        if rc == pb.RC.CONNECTED:
            s3_client = self.manager.get_s3_client(cred)
            bucket_name = request.datasource.bucket
            train = request.datasource.keys.train
            val = request.datasource.keys.validation
            test = request.datasource.keys.test
            
            def load_chunks(l1, l2):
                
                if l1 == 'train':
                    if l2 == "samples":
                        keys = train.samples
                    elif l2 == 'targets':
                        keys = train.targets
                    else:
                        keys = train.manifests
                elif l1 == 'validation':
                    if l2 == "samples":
                        keys = val.samples
                    elif l2 == 'targets':
                        keys = val.targets
                    else:
                        keys = val.manifests
                else:
                    if l2 == "samples":
                        keys = test.samples
                    elif l2 == 'targets':
                        keys = test.targets
                    else:
                        keys = test.manifests
                    
                chunks = []
                for prefix in keys:
                    paginator = s3_client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                    futures = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        for page in pages:
                            if 'Contents' not in page:
                                continue
                            futures.append(executor.submit(self.manager.filter_chunks, page['Contents']))
                    for future in concurrent.futures.as_completed(futures):
                        chunks.extend(future.result())
                chunks.sort(key=lambda x: x['Key'])
                return chunks
            
            # copy data from cloud to NFS, init the `ChunkSize`, `Location`, and `Files` fields
            # collect sample and target etags
            dataset_etags = defaultdict(dict)
            chunks = []
            node_seq = request.nodesequence
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for l1 in ['train', 'validation', 'test']:
                    for l2 in ['samples', 'targets', 'manifests']:
                        raw_chunks = load_chunks(l1, l2)
                        if not raw_chunks:
                            dataset_etags[l1][l2] = []
                            continue
                        else:
                            dataset_etags[l1][l2] = [chunk['ChunkETag'] for chunk in raw_chunks]
                            
                        # downloading ...
                        for chunk in raw_chunks:
                            if not chunk['Exist']:
                                # we also perform data eviction in the clone_chunk function if pre-allocated space is occupied by other jobs
                                futures.append(executor.submit(self.manager.clone_chunk, chunk, s3_client, bucket_name, request.qos.MaxPartMill, node_seq))
                            elif chunk['Location'] != node_seq[0]:
                                # move the data to a node in nodesequence
                                futures.append(executor.submit(self.manager.move_chunk, chunk, node_seq))
                for future in concurrent.futures.as_completed(futures):
                    chunks.extend(future.result())
            
            # write job_col and dataset_col collections
            jobInfo = {
                "Meta": {
                    "Username": cred.username,
                    "JobId": jobId,
                    "Datasource": MessageToDict(request.datasource),
                    "ResourceInfo": MessageToDict(request.resource)
                },
                "QoS": MessageToDict(request.qos),
                "ChunkETags": dict(dataset_etags)
            }
            self.manager.job_col.insert_one(jobInfo)

            for chunk in chunks:
                if not chunk['Exist']:
                    chunk['Jobs'] = [jobId]
                    self.manager.dataset_col.insert_one(chunk)
                else:
                    self.manager.dataset_col.update_one({"ChunkETag": chunk['ChunkETag']}, {"$push": {"Jobs": jobId}})

            return pb.RegisterResponse(rc=pb.RC.REGISTERED, regsucc=pb.RegisterSuccess(
                jobId=jobId,
                mongoUri=self.manager.mongoUri
            ))
        elif rc == pb.RC.DISCONNECTED:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod"))

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jobId
        rc = self.manager.auth_client(cred)
        if rc in [pb.RC.CONNECTED, pb.RC.DISCONNECTED]:
            result = self.manager.job_col.delete_one(filter={"JobId": jobId}) 
            if result.acknowledged and result.deleted_count == 1:
                resp = pb.DeregisterResponse("successfully deregister job {}".format(jobId))
                logger.info('user {} deregister job {}'.format(cred.username, jobId))
            else:
                resp = pb.DeregisterResponse(response='failed to deregister job {}'.format(jobId))
        else:
            resp = pb.DeregisterResponse(response="failed to deregister job {}".format(jobId))
        return resp
      

class DataMissService(pb_grpc.DataMissServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def call(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        if rc != pb.RC.CONNECTED:
            return
        
        print('DataMissService Log: ', request.etag)
        # download data
        def download(etag):
            chunk = self.manager.dataset_col.find_one({"ChunkETag": etag})
            s3_client = self.manager.get_s3_client(cred)
            self.manager.clone_chunk(dataobj=chunk, s3_client=s3_client, bucket_name=chunk['Bucket'], 
                                       chunk_size=chunk['ChunkSize'], node_sequence=[chunk['Location']], part=chunk['Part'], miss=True)
        download(request.etag)
        return pb.DataMissResponse(response=True)


if __name__ == '__main__':
    manager = Manager()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_ConnectionServicer_to_server(ConnectionService(manager), server)
    pb_grpc.add_RegistrationServicer_to_server(RegistrationService(manager), server)
    pb_grpc.add_DataMissServicer_to_server(DataMissService(manager), server)
    server.add_insecure_port(address="{}:{}".format(manager.managerconf['bind'], manager.managerconf['port']))
    server.start()
    server.wait_for_termination()