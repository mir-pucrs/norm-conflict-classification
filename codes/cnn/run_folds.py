import os
import sys
sys.path.append(os.path.abspath('../scripts/'))
import cnn
import read_folds
from preprocess_norms import turn_into_matrix

# Constants.
MAXLENGTH = 200
NB_CLASSES = 2
NB_EPOCH = 100000


def train_cnn(fold, X_train, X_val, y_train, y_val):
    """
        Train CNN network using the data from fold.
    """
    # Get the CNN model.
    model = cnn.get_model(MAXLENGTH)

    save_weights_path = "cnn_model_%s.hdf5" % fold

    checkpointer = ModelCheckpoint(filepath=save_weights_path,
                                 verbose=1,
                                 save_best_only=True
                                 )
    acc_loss_monitor = History()

    early = EarlyStopping(patience=1, verbose=1)

    model.fit_generator(turn_into_matrix(zip(X_train, Y_train), MAXLENGTH),
        samples_per_epoch=166, nb_epoch=NB_EPOCH, verbose=1,
        validation_data=turn_into_matrix(zip(X_val, Y_val), MAXLENGTH),
        nb_val_samples=21, callbacks=[checkpointer, acc_loss_monitor, early])

    val_accs = acc_loss_monitor.history['val_acc']
    val_loss = acc_loss_monitor.history['val_loss']
    
    print val_loss
    sys.exit()

    model.load_weights(save_weights_path)

    score = model.evaluate_generator(turn_into_matrix(zip(X_test, Y_test),
        maxlength), val_samples=21)

    # score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def run_folds(k_fold=False):
    # Setup the Folds class.
    folds = read_folds.Folds(read_folds.FOLDS_PATH,
        read_folds.DATA_PATH[read_folds.ID_CONF],
        read_folds.DATA_PATH[read_folds.ID_N_CONF])

    # Get test set.
    X_test, y_test = folds.read_test()

    # Run over folds.
    for key in folds.structure.keys():

        if key == 'test':
            continue

        print "Processing fold %s" % key
        # Get train and val set corresponding to the fold.
        X_train, X_val, y_train, y_val = folds.read_fold(key)

        # Train CNN for this fold.
        acc, loss, path = train_cnn(key, X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    run_folds()