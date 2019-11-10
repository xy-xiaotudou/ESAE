# -*- encoding: utf8 -*-

import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
data_augmentation = True
num_classes = 10
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

def get_data(Dataset):
    if Dataset == 'cifar10':
        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # # If subtract pixel mean is enabled
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    elif Dataset == 'fashionmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # # If subtract pixel mean is enabled
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1,
                                                                        random_state=42)
    return x_train, x_validation, y_train, y_validation, x_test, y_test
