"""
    Script to split the data set of conflicts into a 10-fold
    cross-validation.
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Constants.
CONF_PATH = '../../data/db_conflicts.csv'
N_CONF_PATH = '../../data/db_non_conflicts_index.csv'
CONF_FILE = '2'
N_CONF_FILE = '1'
N_CONF_CLASS = 0
RAND_STATE = 22
N_SPLITS = 10
TEST_SIZE = 0.1
FILE_PATH = '../../data/10-fold.json'


def process_data(df, conflict=True):
    """
        Extract data and classes from dataframes.
        
        :param df: dataframe containing the csv.
        :type df: pandas.core.frame.DataFrame
        :param conflict: Flag to indicate that df is from a conflict set.
        :type conflict: bool
        :returns: X, y
    """

    X, y = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)

    if conflict:
    
        counter = 0
        # Go through conflict file.
        for i, row in df.iterrows():
            conf_ind = str(row['conflict_id'])
            X[counter] = int(CONF_FILE + conf_ind)
            y[counter] = int(row['conf_type'])
            counter += 1

        return X, y

    else:
        # Go through non-conflict file.
        counter = 0
        rows = list()
        for i, row in df.iterrows():
            n_conf_ind = str(row['index'])
            identifier = int(N_CONF_FILE + n_conf_ind)
            rows.append(identifier)
        
        for i in range(len(df)):
            idn = random.choice(rows)
            X[counter] = idn
            y[counter] = N_CONF_CLASS
            counter += 1
        return X, y


def save_to_json(structure):
    """
        Store the folds structure to a json file.

        :param structure: Dict object containing the divisions between folds.
        :type structure: dict
    """
    try:
        with open(FILE_PATH, 'w') as outfile:
            json.dump(structure, outfile)
        outfile.close()
        print "Correctly saved data to JSON file."
        
    except IOError:
        print('An error occured trying to read the file.')
        
    except ValueError:
        print('Non-numeric data found in the file.')
        
    except KeyboardInterrupt:
        print 'You cancelled the operation.'

    except:
        print "Something went wrong with your JSON file."



def generate_folds(X, y):
    """
        Use X and y to generate a dictionary-like and save as fold division.
        
        :param X: Numpy array containing the corresponding index in the dataset.
        :type X: numpy.ndarray
        :param y: Numpy array containing the corresponding classes.
        :type y: numpy.ndarray
    """

    structure = dict() # Save folds.

    # Divide data into folds and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=TEST_SIZE, random_state=RAND_STATE)

    # Insert test.
    structure['test'] = X_test.tolist()

    # Divide X_train into folds.
    kf = KFold(n_splits=N_SPLITS, random_state=RAND_STATE)
    kf.get_n_splits(X_train)

    fold = 0
    for train_index, test_index in kf.split(X_train):
        structure[fold] = dict()
        structure[fold]['train'] = X_train[train_index].tolist()
        structure[fold]['test'] = X_train[test_index].tolist()
        fold += 1

    print "Folds generated."
    save_to_json(structure)
    

def main():
    
    # Read datasets.
    df = pd.read_csv(CONF_PATH)
    df_2 = pd.read_csv(N_CONF_PATH)

    total_size = len(df) * 2
    X = np.zeros(total_size, dtype=int)
    y = np.zeros(total_size, dtype=int)

    # Get data.
    X[:len(df)], y[:len(df)] = process_data(df)
    
    X_1, y_1 = process_data(df_2, conflict=False)
    X[len(df):total_size], y[len(df):total_size] = X_1[:len(df)], \
        y_1[:len(df)]
    
    # Generate folds.
    generate_folds(X, y)


if __name__ == '__main__':
    main()