import os
import botocore
import time
import boto3
import shutil
import concurrent.futures
import multiprocessing
import socket
import grpc
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc


BANDWIDTH = 100*1e6/8  # 100Mbps


def get_s3_client(s3auth):
    s3_session = boto3.Session(
        aws_access_key_id=s3auth.aws_access_key_id,
        aws_secret_access_key=s3auth.aws_secret_access_key,
        region_name=s3auth.region_name
    )
    s3_client = s3_session.client('s3')
    return s3_client

def get_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

# TODO: 当文件被加压后存在于nfs_storage中，此时怎么判断是否需要重新下载文件
# 考虑在eviction的时候直接删掉整个数据块即使是文件夹，因为无论如何都要重新下载解压
class DownFileService(pb_grpc.DownloadFileServicer):
    
    def call(self, request, context):
        s3auth, bucket, key, dst = request.s3auth, request.bucket, request.key, request.dst
        if os.path.exists(dst) and not os.path.isdir(dst):
            size = os.path.getsize(dst)
            cost = size / BANDWIDTH
        else:
            start = time.time()
            client = get_s3_client(s3auth)
            try:
                client.download_file(bucket, key, dst)
            except botocore.exceptions.ClientError as e:
                return -1
            cost = time.time() - start
            size = os.path.getsize(dst)
        return pb.DownloadFileResponse(size=size, cost=cost)


class ExtractFileService(pb_grpc.ExtractFileServicer):
    
    def call(self, request, context):
        compressed_file = request.compressed_file
        print('extracting file {}...'.format(compressed_file))
        import tarfile, zipfile
        start = time.time()
        if tarfile.is_tarfile(compressed_file) or zipfile.is_zipfile(compressed_file):
            if not os.path.exists(compressed_file):
                os.makedirs(compressed_file)
            src_folder = '{}-tmp'.format(compressed_file)
            if not os.path.exists(src_folder):
                os.mkdir(src_folder)
            os.system('pigz -dc {} | tar xC {}'.format(compressed_file, src_folder))
            os.remove(compressed_file)
            shutil.move(src_folder, compressed_file)
            cost = time.time() - start
        else:
            cost = 0
        return pb.ExtractFileResponse(cost=cost)


if __name__ == '__main__':
    node_ip = os.getenv("NODE_IP")
    assert node_ip is not None
    channel = grpc.insecure_channel("dlcpod-manager:50051")
    worker_join_stub = pb_grpc.WorkerJoinStub(channel)
    join_req = pb.WorkerJoinRequest(node_ip=node_ip, worker_ip=get_ip())
    resp = worker_join_stub.call(join_req)
    
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_DownloadFileServicer_to_server(DownFileService(), server)
    pb_grpc.add_ExtractFileServicer_to_server(ExtractFileService(), server)
    server.add_insecure_port(address="{}:50052".format(get_ip()))
    server.start()
    server.wait_for_termination()