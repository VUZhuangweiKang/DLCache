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
        aws_access_key_id=s3auth['aws_access_key_id'],
        aws_secret_access_key=s3auth['aws_secret_access_key'],
        region_name=s3auth['region_name']
    )
    s3_client = s3_session.client('s3')
    return s3_client

def get_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


class DownFileService(pb_grpc.DownloadFileServicer):
    
    def call(self, s3auth, bucket, key, dst):
        client = get_s3_client(s3auth)
        if os.path.exists(dst):
            size = os.path.getsize(dst)
            cost = size / BANDWIDTH
        else:
            start = time.time()
            try:
                client.download_file(bucket, key, dst)
            except botocore.exceptions.ClientError as e:
                return -1
            cost = time.time() - start
            size = os.path.getsize(dst)
        return size, cost


class ExtractFileService(pb_grpc.ExtractFileServicer):
    
    def call(self, compressed_file):
        import tarfile, zipfile
        start = time.time()
        if tarfile.is_tarfile(compressed_file) or zipfile.is_zipfile(compressed_file):
            if not os.path.exists(compressed_file):
                os.makedirs(compressed_file)
            src_folder = '{}-tmp'.format(compressed_file)
            os.mkdir(src_folder)
            os.system('pigz -dc {} | tar xC {}'.format(compressed_file, src_folder))
            shutil.move(src_folder, compressed_file)
            os.remove(compressed_file)
        else:
            return 0

        cost = time.time() - start
        return cost


if __name__ == '__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_DownloadFileServicer_to_server(DownFileService(), server)
    pb_grpc.add_ExtractFileServicer_to_server(ExtractFileService(), server)
    server.add_insecure_port(address="{}:50052".format(get_ip()))
    server.start()
    server.wait_for_termination()