import sys
import cnn
import argparse
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import History
from keras.callbacks import EarlyStopping
from preprocess_norms import turn_into_matrix
from sklearn.cross_validation import train_test_split

MAX_LEN = 200
CONF = 1
NON_CONF = 0
NB_CLASSES = 2
N_FOLDS = 10
MAX_NON_CONF = 178


def get_bin_pairs(df, conf=True):
    # Get the pairs of binary samples from a dataframe.
    pairs = list()
    for i, row in df.iterrows():
        pairs.append((row['norm1'], row['norm2']))
    if conf:
        y = np.ones(len(df))
    else:
        y = np.zeros(MAX_NON_CONF)

    return pairs[:MAX_NON_CONF], y


def get_mult_pairs(df):
    pairs = list()
    y = np.zeros(len(df))

    for i, row in df.iterrows():
        pairs.append((row['norm1'], row['norm2']))
        if int(row['conf_type']) != 1:
            y[i] = int(row['conf_type']) - 2
        else:
            y[i] = int(row['conf_type']) - 1

    return pairs, y


def train_cnn(model, X, y, maxlength=200):

    # Path to weights.
    save_weights_path = 'models/model_weights.hdf5'
    # Path to accuracies.
    save_accuracies = 'report/accuracies.txt'

    # Divide data into train, validation, and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        random_state=10)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
        test_size=0.5, random_state=11)

    # Set net.
    nb_epoch = 100000

    # X_train = X_train.reshape(X_train.shape[0], maxlength, maxlength, 1)
    # X_val   = X_val.reshape(X_train.shape[0], maxlength, maxlength, 1)
    # X_test  = X_test.reshape(X_test.shape[0], maxlength, maxlength, 1)    

    # print y_train.count(1), y_train.count(0), y_val.count(1), y_val.count(0), y_test.count(1), y_test.count(0)

    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_val = np_utils.to_categorical(y_val, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    checkpointer = ModelCheckpoint(filepath=save_weights_path,
                                 verbose=1,
                                 save_best_only=True
                                 )
    acc_loss_monitor = History()

    early = EarlyStopping(patience=1, verbose=1)

    model.fit_generator(turn_into_matrix(zip(X_train, Y_train), maxlength, NB_CLASSES),
        samples_per_epoch=166, nb_epoch=nb_epoch, verbose=1,
        validation_data=turn_into_matrix(zip(X_val, Y_val), maxlength, NB_CLASSES),
        nb_val_samples=21, callbacks=[checkpointer, acc_loss_monitor, early])

    val_accs = acc_loss_monitor.history['val_acc']
    val_loss = acc_loss_monitor.history['val_loss']

    model.load_weights(save_weights_path)

    if NB_CLASSES == 2:
        w_file = open('prediction_binary.txt', 'w')
    elif NB_CLASSES == 5:
        w_file = open('prediction_multiclass.txt', 'w')

    # w_file.write("Norms\tGold\tPred\n")

    l_text = len(X_test)
    counter = 0

    for m, y in turn_into_matrix(zip(X_test, Y_test), maxlength, NB_CLASSES):

        if counter == l_text:
            break
        pred = model.predict(m)
        pred = np.argmax(pred)
        gold = np.argmax(y)
        w_file.write("Norm\t%d\t%d\n" % (gold, pred))

        counter += 1

    w_file.close()

    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print(score)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])


def prepare_data(model, conf, non_conf, binary=True):
    # Set GPU.
    cnn._start(2)
    conf_len = len(conf)
    non_conf_len = len(non_conf)

    if binary:
        conf_pairs, conf_y = get_bin_pairs(conf)
    else:
        conf_pairs, conf_y = get_mult_pairs(conf)

    non_conf_pairs, non_conf_y = get_bin_pairs(non_conf, conf=False)

    # X = conf_pairs + non_conf_pairs
    # y = np.concatenate((conf_y, non_conf_y))

    train_cnn(model, conf_pairs, conf_y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('conf_path', help='Path to conflict dataset.')
    parser.add_argument('non_conf_path', help='Path to non-conflict dataset.')
    parser.add_argument('-b', '--binary', help="Set 1 for binary")
    args = parser.parse_args()

    if args.binary:
        model = cnn.get_model(MAX_LEN, nb_classes=NB_CLASSES)
    else:
        NB_CLASSES = 5
        model = cnn.get_model(MAX_LEN, nb_classes=NB_CLASSES)

    conflicts = pd.read_csv(args.conf_path)
    non_conflicts = pd.read_csv(args.non_conf_path)
    prepare_data(model, conflicts, non_conflicts, binary=args.binary)