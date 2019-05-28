import pickle 
from matplotlib import pyplot as plt
import os, sys, fnmatch
import numpy as np

def unpickle(path, match):
    res_data = np.ndarray([], dtype=np.uint8)
    res_labels = np.ndarray([])
    res_filenames = np.ndarray([])
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, match):
            with open(path+file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                res_data = np.append(res_data, dict[b'data'])
                res_labels = np.append(res_labels, dict[b'labels'])
                res_filenames = np.append(res_filenames, dict[b'filenames'])
                #print(dict[b'data'])
                print(res_data.size)
                #print(res_data)
    return {'data' : res_data, 'labels' : res_labels, 'filenames' : res_filenames} 

def numpytoimg(dataset, index):
    plt.imshow(dataset['data'].reshape((32, 32, 3), order='F').swapaxes(0,1))
    print(dataset['filenames'][index].decode('utf8'))
    print(dataset['labels'][index])
    plt.show()

def main():
    dict_train = unpickle("./dataset/cifar-10-python/cifar-10-batches-py/", "data_batch_*")
    print("Train Loaded")
    dict_test = unpickle("./dataset/cifar-10-python/cifar-10-batches-py/", "test_batch")
    print("Test Loaded")
#    for x in res.keys():
#        print(x)
    numpytoimg(dict_train, 0)
    print("----------------")
    print(dict_train['filenames'][40000])
    print(dict_train['data'][40000])
main()

