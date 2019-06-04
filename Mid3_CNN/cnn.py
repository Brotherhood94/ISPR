
#printare la confidence sulle prediction
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
import json
import pickle 
import foolbox
from matplotlib import pyplot as plt
import os, sys, fnmatch
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Activation, MaxPool2D, Flatten, Dense 
from keras.models import Sequential, Model, load_model, model_from_json
from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras import utils as np_utils, backend as K, regularizers
from keras.layers.merge import add

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
    img = dataset['data'][index]
    filename = dataset['filenames'][index].decode('utf8')
    label_text = labels_mapping[np.argmax(dataset['labels'][index], axis=0)]
    label_int = dataset['labels'][index]
    fig = plt.figure(filename)
    plt.title(label_text)
    plt.imshow(img)
    plt.show()

def show_attack(image, adversarial, true_label, adversarial_label):
    plt.subplot(1, 3, 1)
    plt.title('Original: '+true_label)
    plt.imshow(image / 255) 
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Difference')
    difference = adversarial - image
    plt.imshow(difference / abs(difference).max()*0.2+0.5)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Adversarial: '+adversarial_label)
    plt.imshow(adversarial / 255) 
    plt.axis('off')

    plt.show()

def identity_block(input_tensor, kernel_size, filters, filters_out=-1):
    norm = BatchNormalization(axis=3)(input_tensor)
    relu = Activation('relu')(norm)
    conv2 = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(relu)
    norm2 = BatchNormalization(axis=3)(conv2)
    relu2 = Activation('relu')(norm2)
    conv3 = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(relu2) 
    add1 = add([input_tensor, conv3])
    norm3 = BatchNormalization()(add1)
    relu3 = Activation('relu')(norm3) 
    if filters_out != -1:
        maxPool = MaxPool2D(pool_size=(2,2), strides=(2,2))(input_tensor)
        conv4 = Conv2D(filters_out, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(maxPool) 
    else:
        conv4 = Conv2D(filters, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(relu3) 
    return conv4 

def load_architecture(path=None):
    with open(path+'model_in_json.json','r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.load_weights(path+'model_weights.h5') 
    return model    

def generate_layers(n_layers, input_tensor, kernel_size, filters):
    for i in range(n_layers):
        x = identity_block(x, kernel_size, filters)
        if i == n_layers-1:
            x = identity_block(x, kernel_size, filters, filters*2)
    return x

def train_architecture(dict_train, dict_test, num_classes, labels_mapping, epochs=None, batch_size=None):
    #Architecture
    n_layers = 4
    input_tensor = Input((32, 32, 3))
    x = Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=dict_train['data'].shape[1:])(input_tensor) 

    generate_layers(n_layers, x, 3, 32) 
    generate_layers(n_layers, x, 3, 64) 
    generate_layers(n_layers, x, 3, 128) 

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.summary()
    opt = optimizers.Adam(lr = 0.0005, decay=1e-4) 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(dict_train['data'], dict_train['labels'], validation_data=(dict_test['data'], dict_test['labels']), epochs=epochs, batch_size=batch_size, verbose=1) 
    loss, acc = model.evaluate(dict_test['data'], dict_test['labels'])
    save_data(model, history, epochs, batch_size, loss, acc, './models')
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    return model

def fast_gradient_sign_method(model, img, label, labels_mapping):
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 255))  

    true_label = labels_mapping[np.argmax(label, axis=0)] 
    predicted_label = labels_mapping[np.argmax(model.predict(np.expand_dims(img, axis=0)))] 
    predicted_label_fmodel = labels_mapping[np.argmax(fmodel.predictions(img))]
    print("true_label: "+str(true_label))
    print("predicted_label: "+str(predicted_label))
    print("predicted_label_fmodel: "+str(predicted_label_fmodel))

    attack = foolbox.attacks.FGSM(fmodel)
    adversarial = attack(img, label=np.argmax(label, axis=0))

    adversarial_label = labels_mapping[np.argmax(fmodel.predictions(adversarial))]
    print("--> adversarial_label: "+str(adversarial_label))

    show_attack(img, adversarial, true_label, adversarial_label) 


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
    dict_train = unpickle_set('./dataset/cifar-10-python/cifar-10-batches-py/', 'data_batch_*', num_classes)
    print('Train Loaded..\n')
    dict_test = unpickle_set('./dataset/cifar-10-python/cifar-10-batches-py/', 'test_batch', num_classes)
    print('Test Loaded..\n')
    print('N° classes: '+str(num_classes))

    #Train Model
#    model = train_architecture(dict_train, dict_test, num_classes, labels_mapping, epochs=sys.argv[1], batch_size=sys.argv[2])
    model = load_architecture('./models/Test_Loss:1.5046686241149902_Test_accuracy:0.5522_256_10/')

    #Load Model
#    model = load_architecture('./models/Test_Loss:1.6799632623672485_Test_accuracy:0.7591_256_80/')

    #FastGradientSign Attack
    fast_gradient_sign_method(model, dict_test['data'][0], dict_test['labels'][0], labels_mapping)

    
main()





