import os
import sys
sys.path.append(os.path.abspath('../scripts/'))
from cnn import *
import read_folds
from preprocess_norms import turn_into_matrix
from preprocess_norms import turn_pair_matrix

# Constants.
MAXLENGTH = 200
NB_CLASSES = 5 
NB_EPOCH = 100000
N_FOLDS = 10
BASE_MODEL_PATH = 'cnn_models/cnn_model_'
K_FOLD = True


def train_cnn(fold, X_train, X_val, y_train, y_val):
    """
        Train CNN network using the data from fold.
    """
    # Get the CNN model.
    model = get_model(MAXLENGTH, nb_classes=NB_CLASSES)

    save_weights_path = BASE_MODEL_PATH + str(fold)

    checkpointer = ModelCheckpoint(filepath=save_weights_path,
                                 verbose=1,
                                 save_best_only=True
                                 )
    acc_loss_monitor = History()

    early = EarlyStopping(patience=1, verbose=1)

    model.fit_generator(turn_into_matrix(zip(X_train, y_train), MAXLENGTH, NB_CLASSES),
        samples_per_epoch=166, nb_epoch=NB_EPOCH, verbose=1,
        validation_data=turn_into_matrix(zip(X_val, y_val), MAXLENGTH, NB_CLASSES),
        nb_val_samples=21, callbacks=[checkpointer, acc_loss_monitor, early])

    val_accs = acc_loss_monitor.history['val_acc']
    val_loss = acc_loss_monitor.history['val_loss']
 	
    last_acc = val_accs[-1]
    last_loss = val_loss[-1]  
    
    print "Model from fold %s obtained accuracy of %.2f and loss of %.2f" % (fold, last_acc, last_loss)

    model.load_weights(save_weights_path)

    return last_acc, last_loss, save_weights_path


def write_results(best_fold, high_acc, min_loss, mean_acc, mean_loss):
	
	# Set the file.
	with open('results.txt', 'w') as outfile:
		message = """CNN Classification\n\nBest fold: %s\nMean Acc: %.2f\nMean Loss: %.2f\nHigh Acc: %.2f\nMin Loss: %.2f\n""" % (best_fold, mean_acc, mean_loss, high_acc, min_loss)
		outfile.write(message)
        print "Just wrote the results file."


def run_test(fold, X_test, y_test, k_fold=False):
	
	# Get model.
	model = get_model(MAXLENGTH, nb_classes=NB_CLASSES)
	# Load weights.
	model.load_weights(BASE_MODEL_PATH + str(fold))

	y_test = np_utils.to_categorical(y_test, NB_CLASSES)

	# Set the output file.
    if k_fold:
        output = open('cnn_test_result_k_fold.txt', 'a')
	else:
        output = open('cnn_test_result.txt', 'w')

	# Run over test set.
	for i in range(len(X_test)):
            matrix = turn_pair_matrix(X_test[i], MAXLENGTH)

            pred = model.predict(matrix)
            pred = np.argmax(pred)
            gold = np.argmax(y_test[i])
            output.write("Norm\t%d\t%d\n" % (gold, pred))
	
	print "Finished test."

 
def run_folds(k_fold=False):
    # Setup the Folds class.
    folds = read_folds.Folds(read_folds.FOLDS_PATH,
    read_folds.DATA_PATH[read_folds.ID_CONF],
    read_folds.DATA_PATH[read_folds.ID_N_CONF])

    if not k_fold:
        # Get test set.
        X_test, y_test = folds.read_test()

    best_fold = None
    high_acc = 0
    min_loss = 1000 
    mean_acc = 0
    mean_loss = 0

    # Run over folds.
    for key in folds.structure.keys():

        if key == 'test':
            continue

        print "Processing fold %s" % key
        # Get train and val set corresponding to the fold.
        X_train, X_val, y_train, y_val = folds.read_fold(key)
        y_train = np_utils.to_categorical(y_train, NB_CLASSES)
        y_val = np_utils.to_categorical(y_val, NB_CLASSES)
        
        # Train CNN for this fold.
        acc, loss, path = train_cnn(key, X_train, X_val, y_train, y_val)

        mean_acc += acc
        mean_loss += loss
	
        if acc > high_acc:
            high_acc = acc
        if loss < min_loss:
            min_loss = loss
            best_fold = key
        elif k_fold:
            run_test(key, X_val, y_val, k_fold=k_fold)
        else:
            # Remove the model.
            os.remove(BASE_MODEL_PATH + str(key))

    mean_acc = mean_acc/N_FOLDS
    mean_loss = mean_loss/N_FOLDS

    write_results(best_fold, high_acc, min_loss, mean_acc, mean_loss)

    if not k_fold:
        run_test(best_fold, X_test, y_test)


if __name__ == '__main__':
    run_folds(k_fold=K_FOLD)
