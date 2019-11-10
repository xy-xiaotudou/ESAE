# -*- encoding: utf8 -*-
import net_model
import tensorflow as tf
import utils
import numpy as np
from sklearn.metrics import accuracy_score
import data

def augmentation(x, img_rows, img_cols, img_channels):
    pad_size = 4
    h, w, c = img_rows, img_cols, img_channels
    pad_h = h + 2 * pad_size
    pad_w = w + 2 * pad_size
    pad_img = np.zeros((pad_h, pad_w, c))
    pad_img[pad_size:h + pad_size, pad_size:w + pad_size, :] = x

    # Randomly crop and horizontal flip the image
    top = np.random.randint(0, pad_h - h + 1)
    left = np.random.randint(0, pad_w - w + 1)
    bottom = top + h
    right = left + w
    if np.random.randint(0, 2):
        pad_img = pad_img[:, ::-1, :]

    aug_data = pad_img[top:bottom, left:right, :]

    return aug_data
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

def get_accuracy(parent_hp, g_name, Dataset):
    if Dataset == 'cifar10':
        img_rows, img_cols, img_channels = 32, 32, 3
    elif Dataset == 'fashionmnist':
        img_rows, img_cols, img_channels = 28, 28, 1

    with g_name.as_default():
        with tf.Session(graph=g_name) as sess:
            x_train, x_validation, y_train, y_validation, x_test, y_test = data.get_data(Dataset=Dataset)
            # Input image dimensions.
            input_shape = (img_rows, img_cols, img_channels)
            model = net_model.test_graph(input_shape, parent_hp, g_name)

            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            loss, accuracy = utils.ModelTrainer(model, x_train, y_train, x_validation, y_validation, 1).train_model()

            scores = model.evaluate(x_test, y_test, verbose=1)
            predictions = []
            for _ in range(100):
                x_test_augmented = np.array([augmentation(image, img_rows, img_cols, img_channels) for image in x_test])
                predictions.append(model.predict(x_test_augmented, verbose=2))
            avg_prediction = np.average(predictions, axis=0)
            y_pred = np.argmax(avg_prediction, axis=1)
            y_true = np.argmax(y_test, axis=1)
            test_accuracy = accuracy_score(y_true, y_pred)
            model.save("./models/my_model" + str(accuracy) + str(scores[1]) + str(test_accuracy) + ".h5")
    return accuracy

def if_pool(layers_hp):
    pool_number = 0
    for item in layers_hp:
        if (len(item) == 3 and item[1] == 2 and ((str(item[2]) == 'max') or (str(item[2]) == 'avg'))) \
                or (len(item) == 4 and item[2] == 2):
            pool_number = pool_number + 1
    if pool_number <= 4:
        pool_flag = 'true'
    else:
        pool_flag = 'false'
    return pool_flag

def is_block(layers_hp):
    is_block = 'false'
    i = -1
    block_id = -1
    for item in layers_hp:
         i += 1
         if len(item) == 3 and item[0] in ['a', 'b']:
            block_id = i
            break
    if block_id == -1:
        is_block = 'true'
    else:
        for j in range(block_id):
            if len(layers_hp[j]) == 4:
                is_block = 'true'
                break
    return is_block

















