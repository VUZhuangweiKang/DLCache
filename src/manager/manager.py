import concurrent.futures
import os, sys
import math
import shutil
import grpc
import boto3
import threading
import json, bson
import time
import configparser
import multiprocessing
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pytz import timezone
import glob
from utils import *


logger = get_logger(name=__name__, level='debug')


class Manager(object):
    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        
        try:
            self.managerconf = parser['manager']
            mconf = parser['mongodb']
        except KeyError as err:
            logger.error(err)
        
        mongo_client = MongoClient(host=mconf['host'], port=int(mconf['port']), username=mconf['username'], password=mconf['password'])
        with open("mongo-schemas/client.json", 'r') as f:
            client_schema = json.load(f)
        with open("mongo-schemas/job.json", 'r') as f:
            job_schema = json.load(f)
        
        # mongo_client.drop_database("Cacher")
        
        collections = mongo_client.Cacher.list_collection_names()
        if 'Client' not in collections:
            self.client_col = mongo_client.Cacher.create_collection(name="Client", validator={"$jsonSchema": client_schema}, validationAction="error")
        else:
            self.client_col = mongo_client.Cacher.Client
            
        if 'Job' not in collections: 
            self.job_col = mongo_client.Cacher.create_collection(name='Job', validator={"$jsonSchema": job_schema}, validationAction="error")
        else:
            self.job_col = mongo_client.Cacher.Job
        
        flush_thrd = threading.Thread(target=Manager.flush_data, args=(self,), daemon=True)
        flush_thrd.start()
        
        logger.info("start global manager")

    def auth_client(self, cred, s3auth=None, conn_check=False):
        result = self.client_col.find_one(filter={"username": cred.username})
        if result is None:
                rc = pb.RC.NO_USER
        else:
            if cred.password == result['password']:
                if conn_check:
                    rc = pb.RC.CONNECTED if result['status'] else pb.RC.DISCONNECTED
                else:
                    rc = pb.RC.CONNECTED
            else:
                rc = pb.RC.WRONG_PASSWORD
        # check whether to update s3auth information
        if rc == pb.RC.CONNECTED and s3auth is not None and result['s3auth'] != s3auth:
            result = self.client_col.update_one(
                filter={"username": cred.username}, 
                update={"$set": {"s3auth": s3auth}})
            if result.modified_count != 1:
                logger.error("user {} is connected, but failed to update S3 authorization information.".format(cred.username))
                rc = pb.RC.FAILED
        return rc
    
    def get_s3_client(self, cred):
        result = self.client_col.find_one(filter={"$and": [{"username": cred.username, "password": cred.password}]})
        s3auth = result['s3auth']
        s3_session = boto3.Session(
            aws_access_key_id=s3auth['aws_access_key_id'],
            aws_secret_access_key=s3auth['aws_secret_access_key'],
            region_name=s3auth['region_name']
        )
        s3_client = s3_session.client('s3')
        return s3_client

    def place_dataset(self, bucket_objs, node_weights):
        n = len(bucket_objs)
        weight_map = {node:0 for node in node_weights}
        node_weights = dict(sorted(node_weights.items(), key=lambda item: item[1], reverse=True))
        nodes = list(node_weights.keys())

        k = 0
        while k < n:
            flag = False
            for i in range(len(nodes)):
                node = nodes[i]
                _, _, free = shutil.disk_usage(node)
                if free > bucket_objs[k]['Size']:
                    bucket_objs[k]['Location'] = node
                    flag = True
                    break
            
            if not flag: # NFS out-of-space
                self.evict_data(node=nodes[0])
            else:
                if weight_map[node] >= math.ceil(n*node_weights[node]): # run out of budget on the i-th node
                    nodes.pop(i)
                k += 1
    
    # evict key from a specific NFS server
    def evict_data(self, node):
        def lru(limit, node):
            """Evict the least N recent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 1.
            """
            
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "Key": "$policy.chunks.Key",
                    "LastAccessTime": "$policy.chunks.LastAccessTime", 
                    "Location": "$policy.chunks.Location"}
                },
                {"$match": {"Location": node}},
                {"$sort": {"LastAccessTime": 1}},
                {"$project": {"Key": 1, "Location": 1}},
                {"$limit": limit}
            ]
            
        def lfu(limit, node):
            """Evict the least N frequent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 1.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "Key": "$policy.chunks.Key",
                    "TotalAccessTime": "$policy.chunks.TotalAccessTime", 
                    "Location": "$policy.chunks.Location"}
                },
                {"$match": {"Location": node}},
                {"$sort": {"TotalAccessTime": 1}},
                {"$project": {"Key": 1, "Location": 1}},
                {"$limit": limit}
            ]
        
        pipeline = {"lru": lru, "lfu": lfu}[self.managerconf['eviction-policy']](limit=self.managerconf.getint('eviction-size'), node=node)
        rmobjs = self.job_col.aggregate(pipeline)
        for obj in rmobjs:
            shutil.rmtree(obj['Location'], ignore_errors=True)
        rmkeys = [obj['Key'] for obj in rmobjs]
        self.job_col.delete_many(
            {"policy.chunks": {"$elemMatch": {"Key": {"$in": rmkeys}}}}
        )

    def clone_s3obj(self, s3obj: dict, s3_client, bucket_name, chunk_size, part=None):
        key = s3obj['Key']
        loc = s3obj['Location']
        s3obj['LastModified'] = int(s3obj['LastModified'].timestamp())
        file_type = key.split('.')[-1].lower()
        if s3obj['Size'] <= chunk_size:
            value = s3_client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
            hash_key = "/{}/{}.{}".format(loc, hashing(value), file_type)
            with open(hash_key, 'wb') as f:
                f.write(value)
            # logger.info("Copy data from s3:{} to dlcache:{}".format(info['Key'], hash_key))
            s3obj['HashKey'] = hash_key
            obj_chunks = [s3obj]
        else:
            s3_client.download_file(Bucket=bucket_name, Key=key, Filename='/tmp/{}'.format(key))
            # logger.info("Download large file s3:{}, size: {}B".format(info['Key'], info['Size']))
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
                        value = chunk.compute()
                        hash_key = "/{}/{}".format(loc, hashing(value))

                        if file_type == 'csv':
                            chunk.to_csv(hash_key, index=False)
                        elif file_type == 'parquet':
                            chunk.to_parquet(path='/', write_metadata_file=False, name_function=lambda _: hash_key)
                        
                        s3obj['Key'] = '{}.part.{}'.format(s3obj['Key'], i)
                        s3obj['HashKey'] = hash_key
                        s3obj['Size'] = os.path.getsize(hash_key)
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
                    while start_row < shape[0]:
                        if part is None or part == p:
                            start_byte = start_row * row_size * dtype.itemsize
                            f.seek(start_byte, 1)
                            if start_row+num_rows > shape[0]:
                                num_rows = shape[0]-start_row+1
                            n_items = row_size * num_rows
                            data = np.fromfile(f, count=n_items, dtype=dtype)
                            data = data.reshape((-1,) + shape[1:])

                            hash_key = "/{}/{}.npy".format(loc, hashing(data))
                            np.save(hash_key)
                            s3obj['Key'] = '{}.part.{}'.format(s3obj['Key'], p)
                            s3obj['HashKey'] = hash_key
                            s3obj['Size'] = sys.getsizeof(data)
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
                            hash_key = "/{}/{}".format(loc, hashing(value))
                            with open(hash_key, 'wb') as f:
                                f.write(value)
                            s3obj['Key'] = '{}.part.{}'.format(s3obj['Key'], p)
                            s3obj['HashKey'] = hash_key
                            s3obj['Size'] = sys.getsizeof(value)
                            obj_chunks.append(s3obj)
                            # logger.info("Copy data from /tmp/{} to dlcache:{}".format(info['Key'], hash_key))
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
                        "username": cred.username,
                        "password": cred.password,
                        "s3auth": MessageToDict(s3auth),
                        "status": True
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
                    "username": cred.username,
                    "password": cred.password,
                },
                update={"$set": {"status": True, "jobs": []}}
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

            # get object keys that are not in MongoDB
            saved_keys = {}
            try:
                saved_job = self.manager.job_col.aggregate([
                    {"$match": {"meta.jobId": jobId}},
                    {"$sort": {"meta.createTime": -1}},
                    {"$limit": 1}
                ]).next()
            except:
                saved_job = None
            if saved_job is not None:
                for chunk in saved_job['policy']['chunks']:
                    saved_keys[chunk['Key']] = chunk
            
            # get bucket objects
            bucket_objs = []
            def list_modified_objects(prefix, page):
                nonlocal bucket_objs
                for info in page['Contents']:
                    info['prefix'] = prefix
                    if saved_job is not None \
                        and info['Key'] in saved_keys \
                        and info['LastModified'].replace(tzinfo=timezone('UTC')).timestamp() == saved_keys[info['Key']]['LastModified']:
                        info['Exist'] = True
                    else:
                        info['Exist'] = False
                    bucket_objs.append(info)
            if len(request.datasource.keys) == 0:
                request.datasource.keys = [bucket_name]
            for prefix in request.datasource.keys:
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for page in pages:
                        futures.append(executor.submit(list_modified_objects, prefix, page))
                concurrent.futures.wait(futures)
                
            self.manager.place_dataset(bucket_objs, request.nodeWeights)
            
            # copy data from S3 to NFS
            obj_chunks = []
            key_lookup = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for obj in bucket_objs:
                    if not obj['Exist']:
                        futures.append(executor.submit(self.manager.clone_s3obj, obj, s3_client, bucket_name, request.qos.maxcachesize))
                    else:
                        obj['HashKey'] = saved_keys[obj['Key']]['HashKey']
                        obj_chunks.append([obj])
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    rlt = future.result()
                    key_lookup[bucket_objs[i]['Key']] = rlt
                    obj_chunks.extend(rlt)
            
            # create job key_lookup in NFS
            with open("/nfs-master/{}/key_lookup.json".format(jobId), 'w') as f:
                json.dump(key_lookup, f)
            
            # save jobinfo to database
            chunks = []
            now = datetime.utcnow().timestamp()
            for ck in obj_chunks:
                ck['TotalAccessTime'] = 0
                ck['LastAccessTime'] = bson.timestamp.Timestamp(int(now), inc=1)
                chunks.append(ck)
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "nodeIP": request.nodeIP,
                    "datasource": MessageToDict(request.datasource),
                    "resourceInfo": MessageToDict(request.resource),
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1)
                },
                "QoS": MessageToDict(request.qos),
                "policy": {
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "chunks": chunks
                }
            }
            result = self.manager.job_col.insert_one(jobInfo)
            if result.acknowledged:
                logger.info('user {} register job {}'.format(cred.username, jobId))
                       
            return pb.RegisterResponse(rc=pb.RC.REGISTERED, regsucc=pb.RegisterSuccess(
                    jinfo=pb.JobInfo(
                        jobId=jobId,
                        createTime=grpc_ts(now)),
                    policy=pb.Policy(key_lookup=list(key_lookup.keys()))))
        elif rc == pb.RC.DISCONNECTED:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod"))

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        rc = self.manager.auth_client(cred)
        if rc in [pb.RC.CONNECTED, pb.RC.DISCONNECTED]:
            result = self.manager.job_col.delete_one(filter={"jobId": jobId}) 
            if result.acknowledged and result.deleted_count == 1:
                resp = pb.DeregisterResponse("successfully deregister job {}".format(jobId))
                logger.info('user {} deregister job {}'.format(cred.username, jobId))
            else:
                resp = pb.DeregisterResponse(response='failed to deregister job {}'.format(jobId))
        else:
            resp = pb.DeregisterResponse(response="failed to deregister job {}".format(jobId))
        return resp
      

class CacheMissService(pb_grpc.CacheMissServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def call(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        if rc != pb.RC.CONNECTED:
            return
        chunk = self.manager.job_col.aggregate([
            {"$project": {
                "_id": 0, 
                "chunk": {
                    "$filter": {
                        "input": "$policy.chunks", 
                        "as": "chunks", 
                        "cond": {"$eq": ["$$chunks.Key", request.key]}}
                    }
                }
            },
            {"$unwind": "$chunk"}
        ]).next()['chunk']
        s3_client = self.manager.get_s3_client(cred)
        key = request.key

        # evict data until the key can be accommodated
        while True:
            _, _, free = shutil.disk_usage(chunk['Location'])
            if free < chunk['Size']:
                self.manager.evict_data(node=chunk['Location'])
            else:
                break
        
        # copy data
        part = None
        if 'part' in key:
            part = int(key.split('.')[1])
        chunk['Key'] = key.split('.')[0]
        self.manager.clone_s3obj(s3obj=chunk, bucket_name=request.bucket, s3_client=s3_client, chunk_size=chunk['Size'], part=part)
        return pb.CacheMissResponse(response=True)


if __name__ == '__main__':
    manager = Manager()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_ConnectionServicer_to_server(ConnectionService(manager), server)
    pb_grpc.add_RegistrationServicer_to_server(RegistrationService(manager), server)
    pb_grpc.add_CacheMissServicer_to_server(CacheMissService(manager), server)
    server.add_insecure_port(address="{}:{}".format(manager.managerconf['bind'], manager.managerconf['port']))
    server.start()
    server.wait_for_termination()