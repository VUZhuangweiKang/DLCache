import numpy as np
import sys


if __name__=="__main__":
    r, c, b = tuple(sys.argv[1:])
    r, c, b = int(r), float(c), int(b)
    dir = "data/run{}/{}/{}".format(r, c, b)
    load_time = np.sum(np.load('{}/load_time.npy'.format(dir)))
    print('Summary: \nload_time(s): {}'.format(load_time))