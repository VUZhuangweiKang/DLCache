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
        wav_files = glob.glob('{}/wav/*'.format(folder))
        txt_files = glob.glob('{}/txt/*'.format(folder))
        wav_files.sort()
        txt_files.sort()
        for i in range(4096):
            futures.append(executor.submit(upload, wav_files[i]))
            futures.append(executor.submit(upload, txt_files[i]))
        concurrent.futures.wait(futures)
        
if __name__ == "__main__":
    upload_objects("train")
    upload_objects("val")