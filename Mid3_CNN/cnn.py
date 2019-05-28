import pickle 
from matplotlib import pyplot as plt
import os, sys, fnmatch
import numpy as np

def unpickle_set(path, match):
    res_data = np.empty((0,3072), np.uint8)
    res_labels = np.empty((0), np.uint8)
    res_filenames = np.empty((0), np.uint8)
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, match):
            with open(path+file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                res_data = np.append(res_data, dict[b'data'], axis=0)
                res_labels = np.append(res_labels, dict[b'labels'])
                res_filenames = np.append(res_filenames, dict[b'filenames'])
    return {'data' : res_data, 'labels' : res_labels, 'filenames' : res_filenames} 


def unpickle_label(file):
    res_mapping = np.empty((0))
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        for label in dict[b'label_names']:
            res_mapping = np.append(res_mapping, label.decode('utf8'))
    return res_mapping 

def numpytoimg(dataset, labels_mapping, index):
    img = dataset['data'][index].reshape((32, 32, 3), order='F')
    filename = dataset['filenames'][index].decode('utf8')
    label_text = labels_mapping[dataset['labels'][index]]
    label_int = dataset['labels'][index]
    fig = plt.figure(filename)
    plt.title(label_text)
    plt.imshow(img.swapaxes(0,1))
    plt.show()

def main():
    dict_train = unpickle_set("./dataset/cifar-10-python/cifar-10-batches-py/", "data_batch_*")
    print("Train Loaded..\n")
    dict_test = unpickle_set("./dataset/cifar-10-python/cifar-10-batches-py/", "test_batch")
    print("Test Loaded..\n")
    labels_mapping = unpickle_label("./dataset/cifar-10-python/cifar-10-batches-py/batches.meta")
    print("Mapping Loaded..\n")
    numpytoimg(dict_train, labels_mapping, 49999)
main()

