import os
import sys
import shutil
import grpc
import time
import boto3, botocore
import json, bson
import configparser
import multiprocessing
import subprocess
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
import concurrent.futures
from collections import defaultdict
from pymongo.mongo_client import MongoClient
from utils import *


logger = get_logger(name=__name__, level='debug')
manager_uri = "dlcpod-manager:50051"

BANDWIDTH = 100*1e6/8  # 100Mbps
API_PRICE = 0.004/1e4
TRANSFER_PRICE = 0.09 # per GB
MAX_CHUNK_SIZE = 1e9
workers = {'129.59.234.236': '10.244.1.3', '129.59.234.237': '10.244.5.6', '129.59.234.238': '10.244.2.63', '129.59.234.239': '10.244.4.6', '129.59.234.241': '10.244.3.6'}

class CHUNK_STATUS:
    PREPARE = 0
    ACTIVE = 1
    PENDING = 2
    COOL_DOWN = 3
    INACTIVE = 4


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
        
        self.download_file_stubs = {}
        self.extract_file_stubs = {}
        for node in workers:
            channel = grpc.insecure_channel('{}:50052'.format(workers[node]))
            self.download_file_stubs[node] = pb_grpc.DownloadFileStub(channel)
            self.extract_file_stubs[node] = pb_grpc.ExtractFileStub(channel)
                        
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

    # def download_file(self, client, bucket, key, etag):
    #     """Download file from S3
    #     Args:
    #         client (Client): s3 client
    #         bucket (str): s3 bucket name
    #         key (str): object key
    #         etag (str): entity tag

    #     Returns:
    #         (str, float, float): destination path, file size, download time
    #     """
    #     tmp_file = '/tmp/{}'.format(etag)
    #     if os.path.exists(tmp_file):
    #         size = os.path.getsize(tmp_file)
    #         cost = size / BANDWIDTH
    #     else:
    #         logger.info("downloading file {} ...".format(key))
    #         start = time.time()
    #         try:
    #             client.download_file(bucket, key, tmp_file)
    #         except botocore.exceptions.ClientError as e:
    #             if e.response["Error"]["Code"] == "404":
    #                 logger.error("Object {} does not exist".format(key))
    #             return -1
    #         cost = time.time() - start
    #         size = os.path.getsize(tmp_file)
    #     return tmp_file, size, cost

    # def extract_file(self, compressed_file):
    #     """Decompress file from src to dst

    #     Args:
    #         src (str): compressed file path
    #         dst (str): decompression folder/file

    #     Returns:
    #         float: time used for extracting the file
    #     """
    #     import tarfile, zipfile
    #     start = time.time()
    #     if tarfile.is_tarfile(compressed_file) or zipfile.is_zipfile(compressed_file):
    #         if not os.path.exists(compressed_file):
    #             os.makedirs(compressed_file)
    #         logger.info("extracting {} ...".format(compressed_file))
    #         src_folder = '{}-tmp'.format(compressed_file)
    #         os.mkdir(src_folder)
    #         os.system('pigz -dc {} | tar xC {}'.format(compressed_file, src_folder))
    #         logger.info("extracted file {} to {} ...".format(compressed_file, src_folder))
    #         shutil.move(src_folder, compressed_file)
    #         os.remove(compressed_file)
    #     else:
    #         return 0

    #     cost = time.time() - start
    #     return cost

    def calculate_cost(self, download_latency, extract_latency, compressed_file_size):
        """Cost = a*(L_download + L_extraction) + (1-a)*(C_API + C_transport)

        Args:
            download_latency (float): time cost of downloading the chunk
            extract_latency (float): time cost of extracting files in the chunk
            compressed_file_size (float):

        Returns:
            float: total eviction cost
        """
        # transport_cost = TRANSFER_PRICE * compressed_file_size
        # cf = float(self.managerconf['costFactor'])
        # total_cost = cf*(download_latency + extract_latency) + (1-cf)*(API_PRICE + transport_cost)
        # return total_cost
        return download_latency + extract_latency

    # Segment a file if it's greater than the max_part_size
    @staticmethod
    def assign_node(node_sequence, chunk_size):
        while True:
            for node in node_sequence:
                free_space = subprocess.check_output("df /%s | awk '{print $4}' | tail -n 1" % node, shell=True)
                free_space = int(free_space) * 1e3
                if free_space >= chunk_size:
                    logger.info("assign file to node {}".format(node))
                    return node

    def seg_tabular_chunk(self, etag, file_type, file_path, node_sequence):
        segments = []
        if file_type in ['csv', 'parquet']:
            from dask import dataframe as DF
            if file_type == 'csv':
                chunks = DF.read_csv(file_path)
            elif file_type == 'parquet':
                chunks = DF.read_parquet(file_path)

            logger.info("repartitioning file {}".format(file_path))
            # this process is time-consuming, depends on the partition size
            chunks = chunks.repartition(partition_size=MAX_CHUNK_SIZE)
            partition_size = chunks.memory_usage_per_partition(deep=True).compute().values
            for part, chunk_part in enumerate(chunks.partitions):
                chunk_etag = "{}-{}".format(etag, part)
                chunk_size = partition_size[part].item()
                loc = self.assign_node(node_sequence, chunk_size)
                seg_path = "/%s/%s" % (loc, chunk_etag)

                logger.info("create partition {}".format(seg_path))
                if not os.path.exists(seg_path):
                    if file_type == 'csv':
                        chunk_part.to_csv(seg_path, index=False, single_file=True)
                    elif file_type == 'parquet':
                        chunk_part.to_parquet(path='/', write_metadata_file=False, name_function=lambda _: seg_path)

                segments.append([chunk_etag, chunk_size, loc])
        elif file_type == 'npy':
            import numpy as np
            with open(file_path, 'rb') as f:
                major, minor = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                assert not fortran, "Fortran order arrays not supported"
                # Get the number of elements in one 'row' by taking a product over all other dimensions.
                row_size = np.prod(shape[1:])
                start_row = 0
                num_rows = MAX_CHUNK_SIZE/(row_size*dtype.itemsize)
                part = 0
                while start_row < shape[0]:
                    start_byte = start_row * row_size * dtype.itemsize
                    f.seek(start_byte, 1)
                    if start_row+num_rows > shape[0]:
                        num_rows = shape[0]-start_row+1
                    n_items = row_size * num_rows
                    value = np.fromfile(f, count=n_items, dtype=dtype)
                    value = value.reshape((-1,) + shape[1:])

                    chunk_etag = "{}-{}".format(etag, part)
                    chunk_size = value.nbytes
                    loc = self.assign_node(node_sequence, chunk_size)
                    seg_path = "/{}/{}".format(loc, chunk_etag)

                    np.save(seg_path, value)

                    segments.append([chunk_etag, chunk_size, loc])
                    start_row += num_rows
                    part += 1
        else:
            with open(file_path, 'rb') as f:
                f.seek(0)
                value = f.read(MAX_CHUNK_SIZE)
                part = 0
                while value:
                    chunk_etag = "{}-{}".format(etag, part)
                    chunk_size = sys.getsizeof(value)
                    loc = self.assign_node(node_sequence, chunk_size)

                    seg_path = "/{}/{}".format(loc, chunk_size)
                    with open(seg_path, 'wb') as f:
                        f.write(value)

                    segments.append([chunk_etag, chunk_size, loc])
                    value = f.read(MAX_CHUNK_SIZE)
                    part += 1
        return segments

    def filter_chunks(self, page):
        """Set the Exist and ChunkETag fields for s3 returned objects

        Args:
            page (List): s3 pages

        Returns:
            List[dict]: a list of s3 object contents
        """
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
            if info['ETag'] not in existing_etags or \
                lastModified > existing_etags[info['ETag']]['LastModified']:
                info['Exist'] = False
            else:
                info = existing_etags[info['ETag']]
                info['Exist'] = True
            info['ChunkETag'] = info['ETag']
            chunks.append(info)
        return chunks

    def acquire_resources(self, chunks, node_sequence):
        """Acquire resources to deploy the chunks

        Args:
            chunks (List[dict]): a list of chunk objects
            node_sequence (List[str]): Node IPs in the cluster

        Returns:
            bool: if the chunks are deployable
        """
        free_space = {}
        for node in node_sequence:
            free_space[node] = shutil.disk_usage("/{}".format(node))[-1]

        remain_chunks = []
        extra_space = 0
        for chunk in chunks:
            s = chunk['Size']
            if chunk['Exist']:
                continue
            if len(free_space) > 0:
                node = list(free_space.items())[0][0]
                if free_space[node] >= s:
                    free_space[node] -= s
                else:
                    remain_chunks.append(chunk)
                    del free_space[node]
            else:
                extra_space += s

        if extra_space > 0:
            # calculate preemptible space
            preempt_space = self.dataset_col.aggregate([
                {"$match": {"Status": CHUNK_STATUS.INACTIVE}},
                {"$group": {"PreemptibleSpace": {"$sum": "$ChunkSize"}}}
            ])['PreemptibleSpace']

            # the job is deployable
            if preempt_space >= extra_space:
                candidates = self.dataset_col.find({'Status': CHUNK_STATUS.INACTIVE})
                extra_space = self.cost_aware_lrfu(candidates, extra_space)
                return extra_space == 0
            else:
                return False
        return True

    def cost_aware_lrfu(self, chunks, require: float):
        scores = []
        for i, chunk in enumerate(chunks):
            t_base, ref_times, cost = datetime.now(), chunk['References'], float(chunk['Cost'])
            crf = 0
            for ref in ref_times:
                dur = (t_base - ref.as_datetime()).total_seconds()
                af = float(self.managerconf['attenuationFactor']) + 1e-9
                crf += cost/dur * pow(1/af, af*dur)
            scores.append((i, crf))
        scores.sort(key=lambda x: x[1])
        rmchunks = []
        for i, _ in scores:
            if require <= 0:
                break
            rmchunks.append(chunks[i]['ChunkETags'])
            path = '/{}/{}'.format(chunks[i]['Location'], chunks[i]['ChunkETag'])
            if os.path.exists(path):
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.remove(path)
        self.dataset_col.delete_many({"ChunkETag": {"$in": rmchunks}})
        if require <= 0:
            return 0
        return require

    # Dataset Rebalance: if some data objects from a dataset is already existing
    def move_chunk(self, chunk, node_sequence):
        while True:
            for node in node_sequence:
                try:
                    if node != chunk['Location']:
                        src_path = '/{}/{}'.format(chunk['Location'], chunk['ChunkETag'])
                        dst_path = '/{}/{}'.format(node, chunk['ChunkETag'])
                        if not os.path.exists(dst_path):
                            shutil.move(src=src_path, dst=dst_path)
                        self.dataset_col.update_one({"ChunkETag": chunk['ChunkETag']}, {"$set": {"Location": node}})
                        chunk['Location'] = node
                    return [chunk]
                except OSError:  # handle the case that the node is out of space
                    continue

    def clone_chunk(self, chunk: dict, s3auth, s3_client, bucket_name, node_sequence):
        key = chunk['Key']
        file_type = key.split('.')[-1].lower()

        if 'ETag' in chunk:
            etag = chunk['ETag'].strip('"')
        else:
            # TODO: this probably overflow memory, we can only read the first 1MB (say)
            value = s3_client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
            etag = hashing(value)

        now = datetime.utcnow().timestamp()
        try:
            last_modified = bson.timestamp.Timestamp(int(chunk['LastModified'].timestamp()), inc=1)
        except:
            last_modified = bson.timestamp.Timestamp(int(chunk['LastModified'].as_datetime().timestamp()), inc=1)
        chunk.update({
            "InitTime": bson.timestamp.Timestamp(int(now), inc=1),
            "LastModified": last_modified
        })

        def extend_chunk_info(raw:dict, chunk_etag, chunk_size, loc):
            cost = self.calculate_cost(download_latency, extract_latency, df_size)
            cost = str((chunk_size/raw['Size']) * cost)
            raw.update({
                "ETag": etag,
                "Bucket": bucket_name,
                "ChunkETag": chunk_etag,
                "ChunkSize": chunk_size,
                "Location": loc,
                "Cost": cost
            })

        # process a chunk which is a file after being downloaded/extracted
        def process_chunk_file(chunk_etag, file_path):
            obj_chunks = []
            if os.path.getsize(file_path) < MAX_CHUNK_SIZE:
                loc = self.assign_node(node_sequence, chunk['Size'])
                extend_chunk_info(chunk, chunk_etag, chunk['Size'], loc)
                obj_chunks.append(chunk)
            else:
                segments = self.seg_tabular_chunk(chunk_etag, file_type, file_path, node_sequence)
                for segment in segments:
                    extend_chunk_info(chunk, *segment)
                    obj_chunks.append(chunk.copy())
                # the old file has been partitioned, so delete it
                os.remove(file_path)
            return obj_chunks

        # download file
        # tmp_path, df_size, download_latency = self.download_file(s3_client, bucket_name, key, etag)
        # actual_file_size = zcat(tmp_path)
        size = chunk['Size']
        loc = self.assign_node(node_sequence, size)
        dst = "/{}/{}".format(loc, etag)
        resp = self.download_file_stubs[loc].call(s3auth, bucket_name, key, dst)
        df_size, download_latency = resp.size, resp.cost
        
        # move compressed file to assigned node
        # shutil.move(tmp_path, dst)

        # extract file
        extract_latency = 0
        if file_type in ['tar', 'bz2', 'zip', 'gz']:
            # extract_latency = self.extract_file(dst)
            extract_latency = self.extract_file_stubs[loc].call(dst).cost

            # walk through extracted files
            if os.path.isdir(dst):
                '''!!! If the extacted entity is a folder, we currently assume any individual
                file in the folder is smaller than the MAX_CHUNK_SIZE
                '''
                extend_chunk_info(chunk, etag, chunk['Size'], loc)
                obj_chunks = [chunk]
            else:
                # compressed chunk (file-based/tabular)
                obj_chunks = process_chunk_file(etag, dst)
        else:
            # uncompressed chunk (file-based/tabular)
            # if not os.path.exists(dst):
            #     shutil.move(tmp_path, dst)
            # else:
            #     os.remove(tmp_path)
            if not os.path.exists(dst):
                shutil.copyfile(tmp_path, dst)
            obj_chunks = process_chunk_file(etag, dst)

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
                {
                    "Username": cred.username,
                    "Password": cred.password,
                },
                {"$set": {"Status": True, "Jobs": []}}
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
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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

            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for l1 in ['train', 'validation', 'test']:
                    raw_chunks  = []
                    for l2 in ['samples', 'targets', 'manifests']:
                        raws = load_chunks(l1, l2)
                        if not raws:
                            dataset_etags[l1][l2] = []
                            continue
                        else:
                            dataset_etags[l1][l2] = [chunk['ChunkETag'] for chunk in raws]
                            for i in range(len(raws)):
                                raws[i]['Category'] = l1
                                if raws[i]["Exist"]:
                                    if raws[i]["Status"]["code"] == CHUNK_STATUS.ACTIVE:
                                        raws[i]["Status"]["active_count"] += 1
                                    else:
                                        raws[i]["Status"]["code"] = CHUNK_STATUS.PENDING
                                else:
                                    raws[i]['Status'] = {"code": CHUNK_STATUS.PENDING, "active_count": 1}
                            raw_chunks.extend(raws)

                    # try to acquire resources for the job
                    if not self.manager.acquire_resources(raw_chunks, node_seq):
                        return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, disk resource is under pressure"))

                    # downloading ...
                    for chunk in raw_chunks:
                        if not chunk['Exist']:
                            futures.append(executor.submit(self.manager.clone_chunk, chunk, cred, s3_client, bucket_name, node_seq))
                        elif chunk['Location'] != node_seq[0]:
                            # move the data to a node in nodesequence
                            futures.append(executor.submit(self.manager.move_chunk, chunk, node_seq))
                            
            for future in concurrent.futures.as_completed(futures):
                chunks.extend(future.result())
            
            # for l1 in ['train', 'validation', 'test']:
            #     raw_chunks  = []
            #     for l2 in ['samples', 'targets', 'manifests']:
            #         raws = load_chunks(l1, l2)
            #         if not raws:
            #             dataset_etags[l1][l2] = []
            #             continue
            #         else:
            #             dataset_etags[l1][l2] = [chunk['ChunkETag'] for chunk in raws]
            #             for i in range(len(raws)):
            #                 raws[i]['Category'] = l1
            #                 if raws[i]["Exist"]:
            #                     if raws[i]["Status"]["code"] == CHUNK_STATUS.ACTIVE:
            #                         raws[i]["Status"]["active_count"] += 1
            #                     else:
            #                         raws[i]["Status"]["code"] = CHUNK_STATUS.PENDING
            #                 else:
            #                     raws[i]['Status'] = {"code": CHUNK_STATUS.PENDING, "active_count": 1}
            #             raw_chunks.extend(raws)

            #     # try to acquire resources for the job
            #     if not self.manager.acquire_resources(raw_chunks, node_seq):
            #         return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, disk resource is under pressure"))

            #     # downloading ...
            #     for chunk in raw_chunks:
            #         if not chunk['Exist']:
            #             chunks.extend(self.manager.clone_chunk(chunk, s3_client, bucket_name, node_seq))
            #         elif chunk['Location'] != node_seq[0]:
            #             # move the data to a node in nodesequence
            #             chunks.extend(self.manager.move_chunk(chunk, node_seq))

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
                if 'References' not in chunk:
                    chunk['References'] = []
                if 'Jobs' not in chunk:
                    chunk['Jobs'] = [jobId]
                else:
                    chunk['Jobs'].append(jobId)

                if not chunk['Exist']:
                    chunk['Exist'] = True
                    self.manager.dataset_col.insert_one(chunk)
                else:
                    self.manager.dataset_col.replace_one({"ChunkETag": chunk['ChunkETag']}, chunk)

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
            self.manager.clone_chunk(chunk=chunk, s3_client=s3_client, bucket_name=chunk['Bucket'], node_sequence=[chunk['Location']])
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