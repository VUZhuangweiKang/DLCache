import concurrent.futures
import os
import sys
import shutil
import grpc
import boto3
import threading
import json, bson
import configparser
import multiprocessing
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from pymongo.mongo_client import MongoClient
from utils import *


logger = get_logger(name=__name__, level='debug')


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

    def filter_objs(self, page):
        try:
            # if prefix is invalid, `Contents` field is not in page, raising error
            etags = {info['ETag']: info['LastModified'] for info in page['Contents']}
        except:
            return []
        results = self.dataset_col.aggregate(pipeline=[
            {"$match": {"ETag": {"$in": list(etags.keys())}}},
            {"$project": {
                "ETag": 1,
                "ChunkETag": 1,
                "LastModified": 1
            }}
        ])
        existing_etags = {
            item['ETag']: {
                "LastModified": item['LastModified'],
                "ChunkETag": item['ChunkETag']
                } for item in results}

        bucket_objs = []
        for info in page['Contents']:
            if info['ETag'] not in existing_etags or info['LastModified'] > existing_etags[info['ETag']]['LastModified']:
                info['Exist'] = False
            else:
                info['Exist'] = True
            bucket_objs.append(info)
        return bucket_objs
    
    # evict keys from a specific NFS server
    def evict_data(self, node):        
        if self.managerconf['eviction-policy'] == 'lru':
            pipeline = [
                {"$match": {"Location": node}},
                {"$sort": {"LastAccessTime": 1}},
                {"$limit": self.managerconf.getint('eviction-size')}]
        elif self.managerconf['eviction-policy'] == 'lru':
            pipeline = [
                {"$match": {"Location": node}},
                {"$sort": {"TotalAccessTime": 1}},
                {"$limit": self.managerconf.getint('eviction-size')}
            ]
        rmobjs = self.job_col.aggregate(pipeline)
        for obj in rmobjs:
            shutil.rmtree('/{}/{}'.format(obj['Location'], obj['ChunkETag']), ignore_errors=True)
        self.job_col.delete_many({"ChunkETag": [obj['ChunkETag'] for obj in rmobjs]})

    def clone_s3obj(self, s3obj: dict, s3_client, bucket_name, chunk_size, node_sequence, part=None):
        key = s3obj['Key']
        s3obj['LastModified'] = bson.timestamp.Timestamp(int(s3obj['LastModified'].timestamp()), inc=1)
        s3obj['TotalAccessTime'] = 0
        now = datetime.utcnow().timestamp()
        s3obj['LastAccessTime'] = bson.timestamp.Timestamp(int(now), inc=1)
        file_type = key.split('.')[-1].lower()

        def schedule_obj(obj):
            while True:
                for node in node_sequence:
                    free_space = shutil.disk_usage("/{}".format(node))[-1]
                    if free_space >= obj['ChunkSize']:
                        obj['Location'] = node
                        return obj
                self.evict_data(node=node_sequence[0])

        # TODO: debug到了这里，会直接进入else，说明chunk_size小于图片大小，bug在/tmp位置，文件不存在
        if s3obj['Size'] <= chunk_size:
            if 'ETag' in s3obj:
                etag = s3obj['ETag']
            else:
                value = s3_client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
                etag = hashing(value)

            s3obj['ChunkETag'] = etag
            s3obj['ChunkSize'] = s3obj['Size']
            s3obj = schedule_obj(s3obj)
            path = "/{}/{}.{}".format(s3obj['Location'], etag, file_type)
            with open(path, 'wb') as f:
                f.write(value)
            obj_chunks = [s3obj]
        else:
            s3_client.download_file(Bucket=bucket_name, Key=key, Filename='/tmp/{}'.format(key))
            obj_chunks = []
            path = '/tmp/{}'.format(key)
            if file_type in ['csv', 'parquet']:
                from dask import dataframe as DF
                if file_type == 'csv':
                    chunks = DF.read_csv(path, index=False)
                elif file_type == 'parquet':
                    chunks = DF.read_parquet(path)

                chunks = chunks.repartition(partition_size=chunk_size)
                for i, chunk in enumerate(chunks.partitions):
                    if part is None or part == i:
                        etag = hashing(chunk.compute())
                        s3obj['ChunkSize'] = chunk.memory_usage_per_partition(deep=True).compute()
                        s3obj = schedule_obj(s3obj)
                        path = "/{}/{}".format(s3obj['Location'], etag)
                        if file_type == 'csv':
                            chunk.to_csv(path, index=False)
                        elif file_type == 'parquet':
                            chunk.to_parquet(path='/', write_metadata_file=False, name_function=lambda _: path)
                        s3obj['ChunkETag'] = etag
                        s3obj['Part'] = i
                        obj_chunks.append(s3obj)
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
                            s3obj['ChunkSize'] = value.nbytes
                            s3obj = schedule_obj(s3obj)
                            path = "/{}/{}".format(s3obj['Location'], etag)
                            np.save(path, value)
                            s3obj['ChunkETag'] = etag
                            s3obj['Part'] = p
                            obj_chunks.append(s3obj)
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
                            s3obj['ChunkSize'] = sys.getsizeof(value)
                            s3obj = schedule_obj(s3obj)
                            path = "/{}/{}".format(s3obj['Location'], etag)
                            with open(path, 'wb') as f:
                                f.write(value)
                            s3obj['ChunkETag'] = etag
                            s3obj['Part'] = p
                            obj_chunks.append(s3obj)
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

            # check whther data objs are exists or out-of-date, init the `Exist` field
            bucket_objs = []
            if len(request.datasource.keys) == 0:
                request.datasource.keys = [bucket_name]
            for prefix in request.datasource.keys:
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for page in pages:
                        futures.append(executor.submit(self.manager.filter_objs, page))
                for future in concurrent.futures.as_completed(futures):
                    bucket_objs.extend(future.result())
            
            # TODO: if (partial) dataset is on the NFS, rebalance dataset across the cluster
            # copy data from S3 to NFS, init the `ChunkSize`, `Location`, and `ChunkTag` fields
            obj_chunks = []
            key_lookup = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for obj in bucket_objs:
                    if not obj['Exist']:
                        futures.append(executor.submit(self.manager.clone_s3obj, obj, s3_client, bucket_name, request.qos.MaxMemoryMill, request.nodeSequence))
                    else:
                        obj_chunks.append([obj])

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    rlt = future.result()
                    key_lookup[bucket_objs[i]['Key']] = rlt
                    obj_chunks.extend(rlt)
            
            # save jobinfo to database
            jobInfo = {
                "Meta": {
                    "Username": cred.username,
                    "JobId": jobId,
                    "Datasource": MessageToDict(request.datasource),
                    "ResourceInfo": MessageToDict(request.resource)
                },
                "QoS": MessageToDict(request.qos),
                "ETags": [item['ChunkETags'] for item in obj_chunks]
            }
            self.manager.job_col.insert_one(jobInfo)

            # save dataset info into database
            for chunk in obj_chunks:
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
        jobId = request.jinfo.jobId
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
        chunk = self.manager.dataset_col.find_one({"ETag": request.etag})
        s3_client = self.manager.get_s3_client(cred)

        # evict data until the key can be accommodated
        while True:
            _, _, free = shutil.disk_usage("/{}".format(chunk['Location']))
            if free < chunk['ChunkSize']:
                self.manager.evict_data(node=chunk['Location'])
            else:
                break

        # copy data
        self.manager.clone_s3obj(s3obj=chunk, bucket_name=request.bucket, s3_client=s3_client, chunk_size=chunk['ChunkSize'], part=chunk['Part'])
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