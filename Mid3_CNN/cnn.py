#https://medium.com/@eternalzer0dayx/demystifying-convolutional-neural-networks-ca17bdc75559
#Multiple Layer: 1 edeges, 2 shapes, 3 collction of edges, 4 collection  of shapes (es truck)
#BUT Regularization becomes important as the number of parameters (weights) increase to avoid memorization and helping generalization of the feautures

#https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
#il dropout non si usa ne conv ma solo nei dense. Basta il batchnorm nei conv (dim screenshot)
#fare un confronto in 8 epoche con dropout e basth norm (screen)
#Output layes is a dense layer of 10 nodes (as there are 10 classes woth softmax
import pickle 
from matplotlib import pyplot as plt
import os, sys, fnmatch
import numpy as np
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, Dropout, Conv2D, Activation, MaxPool2D, Flatten, Dense 
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
from keras import utils as np_utils, backend as K

def unpickle_set(path, match, num_classes):
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
    return {'data' : res_data.reshape((len(res_data), 3, 32, 32)).transpose(0,2,3,1), 'labels' : np_utils.to_categorical(res_labels, num_classes=num_classes), 'filenames' : res_filenames} 


def unpickle_label(file):
    res_mapping = np.empty((0))
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        for label in dict[b'label_names']:
            res_mapping = np.append(res_mapping, label.decode('utf8'))
    return res_mapping 

def plot_data(history, metric, epochs, color1, color2):
    plt.clf()
    z = history.history[metric]
    val_z = history.history['val_'+metric]
    plt.plot(range(1,epochs+1), z, color=color1, label='Traning '+metric.capitalize())
    plt.plot(range(1,epochs+1), val_z, color=color2, label='Validation '+metric.capitalize())
    plt.title('Traning/Validation '+metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    return plt

def save_data(model, history, epochs, batch_size, loss, acc, path):
    path = path+'/Test_Loss:'+str(loss)+'_Test_accuracy:'+str(acc)+'_'+str(batch_size)+'_'+str(epochs)+'/'
    try:  
        os.mkdir(path)
    except OSError:  
        print ('Creation of the directory %s failed' % path)
    else:  
        print ('Successfully created the directory %s ' % path)
    model.save_weights(path+'model.h5')
    plot_data(history, 'loss', epochs, 'orange', 'blue').savefig(path+'loss.png')
    plot_data(history, 'acc', epochs, 'red', 'green').savefig(path+'acc.png')
    plot_model(model, to_file=path+'model_plot.png', show_shapes=True, show_layer_names=True)

def numpytoimg(dataset, labels_mapping, index):
    plt.clf()
    img = dataset['data'][index]
    filename = dataset['filenames'][index].decode('utf8')
    label_text = labels_mapping[np.argmax(dataset['labels'][index], axis=0)]
    label_int = dataset['labels'][index]
    fig = plt.figure(filename)
    plt.title(label_text)
    plt.imshow(img)
    plt.show()


def main():
    #GPU checkup 
    print("GPU info")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)

    #Load Dataset 
    labels_mapping = unpickle_label('./dataset/cifar-10-python/cifar-10-batches-py/batches.meta')
    num_classes = len(labels_mapping)
    print('Mapping Loaded..\n')
    print('N° classes: '+str(num_classes))
    dict_train = unpickle_set('./dataset/cifar-10-python/cifar-10-batches-py/', 'data_batch_*', num_classes)
    print('Train Loaded..\n')
    dict_test = unpickle_set('./dataset/cifar-10-python/cifar-10-batches-py/', 'test_batch', num_classes)
    print('Test Loaded..\n')
    #numpytoimg(dict_train, labels_mapping, 49999)

    #Architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=dict_train['data'].shape[1:]))
    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
    model.add(BatchNormalization()) 
    
    model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization()) 
    
    model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(3,3), strides=(1,1))) 
    model.add(BatchNormalization()) 
    
    model.add(Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(1,1), strides=(1,1)))  
    model.add(BatchNormalization()) 
    
    model.add(Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(1,1), strides=(1,1)))  
    model.add(BatchNormalization()) 
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    batch_size = 256 
    epochs = 100 
    history = model.fit(dict_train['data'], dict_train['labels'], validation_data=(dict_test['data'], dict_test['labels']), epochs=epochs, batch_size=batch_size, verbose=1) 

    loss, acc = model.evaluate(dict_test['data'], dict_test['labels'])

    print('Test loss:', loss)
    print('Test accuracy:', acc)
    save_data(model, history, epochs, batch_size, loss, acc, './models')


main()

