# Larger CNN for the MNIST Dataset
import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adamax
from keras.optimizers import Adam
from os import listdir
from os.path import isfile, join

# define the small model
def small_model():
    # 75.59%
    # create model
    model = Sequential()
    model.add(Conv2D(15, (3, 3), input_shape=(1, 28, 28), activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(num_classes, activation='sigmoid'))
    # model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the larger model
def larger_model():
    # 77%
    # 6 99%
    # 10 80%
    # 15 76%
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def segmentation_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(28*28, activation='relu'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def very_big_model():
    # 6  0.8%
    # 16 1.0%
    # 31
    # create model
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(num_classes*100, activation='relu'))
    model.add(Dense(num_classes*10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def full_cnn_model():
    # 1.36% 50
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes*100, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def full_cnn_revert_model():
    # 1.93% 6
    # 1.37% 20
    # create model
    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(60, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes*100, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
    return model

def full_cnn_1_model():
    # 1.93% 6
    # 1% 6 metrics=['categorical_accuracy'] optimizer='adamax' loss='mean_squared_logarithmic_error'
    # 1.09% 6 loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy']
    # 1.21% 6 loss = 'mean_squared_logarithmic_error', optimizer = adamaxDecay, metrics = ['accuracy'] decay=1e-6 batch_size=120
    # 1.27% 6 loss = 'mean_squared_logarithmic_error', optimizer = adamDecay, metrics = ['accuracy'] decay=1e-6 batch_size=240
    # create model
    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes*100, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    adamaxDecay = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=adamaxDecay, metrics=['accuracy'])
    return model


def big_cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(60, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
    return model

def compile_model(model, file_name, train, train_lable, test, test_lable, epochs=10):
    # Fit the model
    model.fit(train, train_lable, validation_data=(test, test_lable), epochs=epochs, batch_size=240)
    # Final evaluation of the model
    scores = model.evaluate(test, test_lable, verbose=0)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save(file_name)  # creates a HDF5 file 'my_model.h5'

def load_simple():
    lable = []
    for i in [0,1,2,3,4,5,6,7,7,8,9]:
        lable_i = np.zeros(10)
        lable_i[i] = 0.95
        lable.append(lable_i)

    dir = "../../assets/basic-numbers/"
    CV_LOAD_IMAGE_GRAYSCALE = 0
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    images = [(255 - cv2.imread(dir + path, CV_LOAD_IMAGE_GRAYSCALE))/255 for path in onlyfiles[1:]]

    return np.array(images),np.array(lable)

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

simple_image, simple_label = load_simple()
simple_image = simple_image.reshape(simple_image.shape[0], 1, 28, 28).astype('float32')

# # build the model
# model = larger_model()
# file_name = 'mnist_cnn_model.h5'
# compile_model(model, file_name, 10)

# build the model
# model = larger_model()
# file_name = 'mnist_cnn_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)

# build the model
# model = big_cnn_model()
# file_name = 'mnist_big_cnn_adam_20_model.h5'
# model = load_model('mnist_big_cnn_adam_10_model.h5')
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)
# file_name = 'mnist_big_cnn_adam_30_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)
# file_name = 'mnist_big_cnn_adam_40_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)
# file_name = 'mnist_big_cnn_adam_50_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)
# file_name = 'mnist_big_cnn_adam_60_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 10)
# file_name = 'mnist_big_cnn_adam_20_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 20)
# file_name = 'mnist_big_cnn_adam_30_model.h5'
# compile_model(model, file_name, X_train, y_train, X_test, y_test, 30)


Xtag_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')
Xtag_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')
model = segmentation_model()
file_name = 'mnist_segmentation_model.h5'
compile_model(model, file_name, X_train, Xtag_train, X_test, Xtag_test, 5)