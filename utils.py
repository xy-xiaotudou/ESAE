import os
import pickle
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import constant

subtract_pixel_mean = True
def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 375:
        lr = 1e-3
    elif epoch > 250:
        lr = 1e-2
    elif epoch > 5:
        lr = 1e-1
    return lr


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message

class ModelTrainer:
    """A class that is used to train model

    This class can train a model with dataset and will not stop until getting minimum loss

    Attributes:
        model: the model that will be trained
        x_train: the input train data
        y_train: the input train data labels
        x_test: the input test data
        y_test: the input test data labels
        verbose: verbosity mode
    """

    def __init__(self, model, x_train, y_train, x_test, y_test, verbose):
        """Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose"""
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose

    def train_model(self,
                    max_iter_num=constant.MAX_ITER_NUM,
                    batch_size=constant.MAX_BATCH_SIZE,
                    optimizer=None,
                    augment=constant.DATA_AUGMENTATION):
        """Train the model.

        Args:
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.
            batch_size: An integer. The batch size during the training.
            optimizer: An optimizer class.
            augment: A boolean of whether the data will be augmented.
        """
        if augment:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)
            datagen.fit(self.x_train)
        else:
            datagen = None
        if optimizer is None:
            self.model.compile(loss=categorical_crossentropy,
                               optimizer=Adam(lr=lr_schedule(0)),
                               metrics=['accuracy'])
        else:
            self.model.compile(loss=categorical_crossentropy,
                               optimizer=optimizer(),
                               metrics=['accuracy'])

        batch_size = min(self.x_train.shape[0], batch_size)

        try:
            if augment:
                flow = datagen.flow(self.x_train, self.y_train, batch_size)

                results = self.model.fit_generator(flow, steps_per_epoch=self.x_train.shape[0]//batch_size,
                                        epochs=max_iter_num,
                                        validation_data=(self.x_test, self.y_test),
                                        validation_steps=self.x_test.shape[0]//batch_size,
                                        verbose=self.verbose)
            else:
                results = self.model.fit(self.x_train, self.y_train,
                               batch_size=batch_size,
                               epochs=max_iter_num,
                               validation_data=(self.x_test, self.y_test),
                               verbose=self.verbose)
        except NoImprovementError as e:
            if self.verbose:
                print('Training finished!')
                print(e.message)

        return results.history["val_loss"], results.history["val_acc"][-1]

def ensure_dir(directory):
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_file_dir(path):
    """Create path if it does not exist"""
    ensure_dir(os.path.dirname(path))

def has_file(path):
    return os.path.exists(path)

def pickle_from_file(path):
    return pickle.load(open(path, 'rb'))

def pickle_to_file(obj, path):
    pickle.dump(obj, open(path, 'wb'))


