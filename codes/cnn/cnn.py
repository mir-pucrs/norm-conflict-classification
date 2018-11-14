"""
    GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py    
"""
import csv
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import History
from keras.callbacks import EarlyStopping

from constants import *
from preprocess_norms import turn_into_matrix

np.random.seed(789)


# Set the gpu to use.
def _start(gpu):
    import os
    import tensorflow as tf

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

_start(2)


def as_keras_metric(method):
    # Method that allow using Tensorflow metrics on Keras.
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def train_cnn(maxlength=200):

    # Get model.
    model = get_model(maxlength)

    # Get data.
    X, y = select_data()

    # Path to weights.
    save_weights_path = 'models/model_weights.hdf5'
    # Path to accuracies.
    save_accuracies = 'report/accuracies.txt'

    # Divide data into train, validation, and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
        test_size=0.5, random_state=42)

    # Set net.
    nb_classes = 2
    nb_epoch = 100000

    # X_train = X_train.reshape(X_train.shape[0], maxlength, maxlength, 1)
    # X_val   = X_val.reshape(X_train.shape[0], maxlength, maxlength, 1)
    # X_test  = X_test.reshape(X_test.shape[0], maxlength, maxlength, 1)    

    # print y_train.count(1), y_train.count(0), y_val.count(1), y_val.count(0), y_test.count(1), y_test.count(0)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    checkpointer = ModelCheckpoint(filepath=save_weights_path,
                                 verbose=1,
                                 save_best_only=True
                                 )
    acc_loss_monitor = History()

    early = EarlyStopping(patience=1, verbose=1)

    model.fit_generator(turn_into_matrix(zip(X_train, Y_train), maxlength),
        samples_per_epoch=166, nb_epoch=nb_epoch, verbose=1,
        validation_data=turn_into_matrix(zip(X_val, Y_val), maxlength),
        nb_val_samples=21, callbacks=[checkpointer, acc_loss_monitor, early])

    val_accs = acc_loss_monitor.history['val_acc']
    val_loss = acc_loss_monitor.history['val_loss']

    model.load_weights(save_weights_path)

    score = model.evaluate_generator(turn_into_matrix(zip(X_test, Y_test),
        maxlength), val_samples=21)

    # score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def select_data():

    conflict_path = BASE_FOLDER_PATH + "conflicts.csv"
    non_conflict_path = BASE_FOLDER_PATH + "non-conflicts.csv"

    conflicts, non_conflicts = [], []

    with open(conflict_path, 'r') as csvfile:

        rdr = csv.reader(csvfile, delimiter=',')

        for row in rdr:

            conflicts.append(((row[0], row[1]), 1))

    with open(non_conflict_path, 'r') as csvfile:

        rdr = csv.reader(csvfile, delimiter=',')

        for row in rdr:

            non_conflicts.append(((row[0], row[1]), 0))

    np.random.shuffle(non_conflicts)

    n_conflicts = len(conflicts)

    print n_conflicts

    non_conflicts = non_conflicts[:n_conflicts]

    print len(conflicts), len(non_conflicts)

    total_data = conflicts + non_conflicts

    X, y = [x for x, y in total_data], [y for x, y in total_data]

    return X, y


def get_model(maxlength, nb_classes=2):

    # input image dimensions
    img_rows, img_cols = maxlength, maxlength
    input_shape = (img_rows, img_cols, 1)

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    print model.input_shape, model.output_shape
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    print model.input_shape, model.output_shape
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    print model.input_shape, model.output_shape
    model.add(Dropout(0.25))

    model.add(Flatten())
    print model.input_shape, model.output_shape
    model.add(Dense(128))
    print model.input_shape, model.output_shape
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', precision, recall])

    return model

if __name__ == '__main__':

    train_cnn()
    # model = get_model(200)
