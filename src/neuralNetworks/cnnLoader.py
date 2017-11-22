from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils


def MNIST_CNN_Model():
    return load_model('../data/mnist_cnn_model.h5')


def MNIST_CNN_sigmoid_Model():
    return load_model('../data/mnist_small_cnn_model_sigmoid.h5')


def MNIST_CNN_simple_Model():
    return load_model('../data/mnist_simple_cnn_model.h5')


def MNIST_CNN_adamax_Model():
    return load_model('../data/mnist_full_cnn_adamax_5_model.h5')


def MNIST_CNN_adamax_2_Model():
    return load_model('../data/mnist_full_cnn_adamax_5_2_model.h5')


def MNIST_CNN_adamax_3_Model():
    return load_model('../data/mnist_full_cnn_adamax_5_3_model.h5')


def MNIST_CNN_adamax_4_Model():
    return load_model('../data/mnist_full_cnn_adamax_5_4_model.h5')

def MNIST_CNN_adam_6_Model():
    return load_model('../data/mnist_full_cnn_adam_6_model.h5')


def MNIST_CNN_adam_16_Model():
    return load_model('../data/mnist_full_cnn_adam_16_model.h5')


def MNIST_CNN_adam_31_Model():
    return load_model('../data/mnist_full_cnn_adam_31_model.h5')


def MNIST_CNN_adam_1_Model():
    return load_model('../data/mnist_cnn_adam_6_model.h5')


def MNIST_CNN_adam_2_Model():
    return load_model('../data/mnist_cnn_adam_10_model.h5')


def MNIST_CNN_adam_3_Model():
    return load_model('../data/mnist_cnn_adam_15_model.h5')


def MNIST_CNN_big_adam_3_Model():
    return load_model('../data/mnist_big_cnn_adam_10_model.h5')


def MNIST_segmentation_Model():
    return load_model('../data/mnist_segmentation_model.h5')


def MNIST_test_model(model):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_test = X_test / 255
    y_test = np_utils.to_categorical(y_test)

    scores = model.evaluate(X_test, y_test, verbose=0, batch_size=240)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))