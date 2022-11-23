from PIL import Image
import time
import glob
import numpy as np
import pandas as pd
import json
import io
import os
from argparse import Namespace
import concurrent
import concurrent.futures

NUM_THREADS = 12
BUCKET_NAME  = "vuzhuangwei"

def reader(path):
    t0 = time.time()
    Image.open(path)
    load_time = time.time() - t0
    return load_time

def s3_read(client, key):
    t0 = time.time()
    data = client.get_object(Bucket=BUCKET_NAME, Key=key)
    Image.open(io.BytesIO(data))
    load_time = time.time() - t0
    return load_time

def read_secret(key):
    path = '/secret/{}'.format(key)
    assert os.path.exists(path)
    with open(path, 'r') as f:
        data = f.read().strip()
    return data


def main(config):
    args = Namespace(**config)
    if not args.enable_fscache:
        os.system("vmtouch -e /hdd/")
    futures = []
    total_size = 0
    if args.remote:
        import boto3
        import configparser
        config = configparser.ConfigParser()
        config.read("/secret/credentials")
        session = boto3.Session(
            aws_access_key_id=read_secret('aws_access_key_id'),
            aws_secret_access_key=read_secret('aws_secret_access_key'),
            region_name=read_secret('region_name')
        )
        client = session.client('s3')
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=args.dataset)
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            for page in pages:
                for obj in page['Contents']:
                    total_size += obj["Size"]
                    if total_size > 1e9 * args.read_size:
                        break
                    futures.append(executor.submit(reader, obj['Key']))
    else:
        files = glob.glob(args.dataset)
        np.random.shuffle(files)
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            for file in files:
                total_size += os.path.getsize(file)
                if total_size > 1e9 * args.read_size:
                    break
                futures.append(executor.submit(reader, file))
    
    results = []
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result()) 
    results = np.array(results)
    pd.DataFrame(results).describe()
    return results
    

if __name__ == '__main__':
    with open("configs.json") as f:
        configs = json.load(f)
    for i, config in enumerate(configs):
        print('start evaluating config {}'.format(i))
        rlt = main(config)
        np.save('./data/exp_{}'.format(i), rlt)