from PIL import Image
import glob
import torchvision.transforms as transforms
import time
import numpy as np
import multiprocessing
import threading


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

def main(args):
    files = glob.glob("/129.59.234.238/*")
    np.random.shuffle(files)
    summary = []
    for path in files[:1000]:
        t = time.time()
        img = Image.open(path)
        img = img.convert("RGB")
        img = transform(img)
        dur = time.time() - t
        print(dur)
        summary.append(dur)
    print(np.mean(summary), args)
        
        
if __name__ == '__main__':
    main('default')
    proc = multiprocessing.Process(target=main, args=('in-process', ), daemon=True)
    proc.start()
    # proc.join()

    # thread = threading.Thread(target=main, args=('in-thread', ), daemon=True)
    # thread.start()
    # thread.join()
    
    while True:
        pass