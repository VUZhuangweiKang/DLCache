# This file should be placed in the ~/LibriSpeech/LibriSpeech_dataset foler


import boto3
import glob
import multiprocessing
import concurrent.futures

session = boto3.Session()
s3 = session.client("s3")
bucket = 'vuzhuangwei'

def upload_objects(folder):
    upload = lambda path: s3.upload_file(path, bucket, 'LibriSpeech/{}'.format(path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        imgs = glob.glob('{}/*/*'.format(folder))
        for path in imgs:
            futures.append(executor.submit(upload, path))
        concurrent.futures.wait(futures)
        
if __name__ == "__main__":
    upload_objects("train")
    upload_objects("val")