import boto3
import glob
from PIL import Image
import pickle
import multiprocessing
import concurrent.futures
import torchvision.transforms as transforms


session = boto3.Session()
s3 = session.client("s3")
bucket = 'vuzhuangwei'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def preprocess(path):
    key = 'Imagenet-Mini-Obj/{}'.format(path)
    img = Image.open(path)
    img = img.convert("RGB")
    img = transform(img)
    s3.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(img))
    return key
 
def upload_objects(folder):
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        imgs = glob.glob('{}/*/*'.format(folder))
        for path in imgs:
            futures.append(executor.submit(preprocess, path))
        concurrent.futures.wait(futures)

            
if __name__=="__main__":
    upload_objects('train')
    upload_objects('val')