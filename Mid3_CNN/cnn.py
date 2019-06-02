#It turns out that 1×1 conv can reduce the number of connections (parameters) while not degrading the performance of the network so much
#éiù profonda la voglo fare, più problema del vanishing
#Max pooling extracts the most important features like edges whereas, average pooling extracts features so smoothly. For image data, you can see the difference. Although both are used for same reason, I think max pooling is better for extracting the extreme features. 
#In 2014, Springenber et al. published a paper entitled Striving for Simplicity: The All Convolutional Net which demonstrated that replacing pooling layers with strided convolutions can increase accuracy in some situations.
#You may use dilated convolution when:You are working with higher resolution images but fine-grained details are still important
#https://medium.com/@eternalzer0dagg/demystifying-convolutional-neural-networks-ca17bdc75559
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
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Activation, MaxPool2D, Flatten, Dense 
from keras.models import Sequential, Model, load_model, model_from_json
from keras.utils.vis_utils import plot_model
from keras import utils as np_utils, backend as K, regularizers
from keras.layers.merge import add
import json

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
    model_json = model.to_json()
    with open(path+"model_in_json.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights(path+'model_weights.h5')
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


def show_intermediate_activation(model, dict_test):
    layer_names = []
    layer_name = None
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs) 
    img = np.expand_dims(dict_test['data'][0], axis=0)
    activations = activation_model.predict(img)[:12]
    print(len(activations))
    for layer in layer_outputs[:12]:
        layer_names.append(layer.name)
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        print(n_features//images_per_row)
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        print(display_grid)
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                        :, :,
                        col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                    row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def load_architecture(dict_test, labels_mapping):
    path = "./models/Test_Loss:1.5046686241149902_Test_accuracy:0.5522_256_10/"
    with open(path+'model_in_json.json','r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.load_weights(path+'model_weigths.h5') #model_weights.h5
    show_intermediate_activation(model, dict_test)
'''
    layer_name = None
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs) 
    img = np.expand_dims(dict_test['data'][0], axis=0)
    print(img.shape)
    activations = activation_model.predict(img) 
    #print(activations)
    #print(labels_mapping[np.argmax(activations, axis=1)])
    #numpytoimg(dict_test, labels_mapping, 0)\:w
    first_layer_activation = activations[4]
    print(first_layer_activation.shape)
    plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis')
    plt.show()
'''

def identity_block(input_tensor, kernel_size, filters, filters_out=-1):
    norm = BatchNormalization(axis=3)(input_tensor)
    relu = Activation('relu')(norm)
    conv2 = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(relu)
    norm2 = BatchNormalization(axis=3)(conv2)
    relu2 = Activation('relu')(norm2)
    conv3 = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(relu2) 
    add1 = add([input_tensor, conv3])
    norm3 = BatchNormalization(axis=3)(conv3)
    relu3 = Activation('relu')(norm3) 
    if filters_out != -1:
        conv4 = Conv2D(filters_out, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(input_tensor) 
    else:
        conv4 = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(relu3) 
    return conv4 

def main():
    #GPU checkup 
    print("GPU info")
    config = tf.ConfigProto()
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
    save_architecture(dict_train, dict_test, num_classes, labels_mapping)
    #load_architecture(dict_test, labels_mapping)

def save_architecture(dict_train, dict_test, num_classes, labels_mapping):
    #Architecture
    input_tensor = Input((32, 32, 3))
    x = Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=dict_train['data'].shape[1:])(input_tensor) 
    x = identity_block(x, 3, 32, 64)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = identity_block(x, 3, 64, 128)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = identity_block(x, 3, 128, 256)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = identity_block(x, 3, 256)
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)

#    model = Sequential()
#    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=dict_train['data'].shape[1:]))
#    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
#    model.add(BatchNormalization()) 
#    
#    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
#    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
#    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))  
#    model.add(BatchNormalization()) 
#    
#    model.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), activation='relu'))
#    model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu'))
#    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))  
#    model.add(BatchNormalization()) 
#     
#    model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
#    model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
#    model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))  
#    model.add(BatchNormalization()) 
    
    
#    model.add(Flatten())
#    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.4))
#    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    batch_size = 256 
    epochs = 135 
    history = model.fit(dict_train['data'], dict_train['labels'], validation_data=(dict_test['data'], dict_test['labels']), epochs=epochs, batch_size=batch_size, verbose=1) 

    loss, acc = model.evaluate(dict_test['data'], dict_test['labels'])

    print('Test loss:', loss)
    print('Test accuracy:', acc)
    save_data(model, history, epochs, batch_size, loss, acc, './models')


main()

