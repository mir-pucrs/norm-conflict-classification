import sys
import json
import numpy as np
import pandas as pd

# Constants.
FOLDS_PATH = '../../data/K-fold.json'
DATA_PATH = {'2': '/usr/share/datasets/conflicts/db_conflicts.csv',
             '1': '/usr/share/datasets/conflicts/db_non_conflicts.csv'}
ID_CONF = '2'
ID_N_CONF = '1'
CONF_CLASS = 1
N_CONF_CLASS = 0
BINARY = True


class Folds():
    """Read and process the 10-fold structure."""
    
    def __init__(self, path, conf_path, n_conf_path, binary):
        self.path = path
        self.structure = json.loads(open(path, 'r').read())
        self.conf_path = conf_path
        self.n_conf_path = n_conf_path
        self.dfs = dict()
        self.dfs[ID_CONF] = pd.read_csv(self.conf_path)
        self.dfs[ID_N_CONF] = pd.read_csv(self.n_conf_path)
        self.binary = binary

    def read_test(self):
        """Return the test set from structure."""
        
        ids_list = self.structure['test']
        X, Y = [], []

        for elem_id in ids_list:
            x, y = self.read_id(elem_id)
            X.append(x)
            if not self.binary:
                if y > 1:
                    y = y - 1 
            Y.append(y)

        return X, Y

    def read_fold(self, fold):
        """
            Return a train/test structure corresponding to the fold.

            :param fold: Fold number.
            :type fold: int
        """
        assert fold in self.structure, "Structure doesn't have fold %d" % fold

        train = self.structure[fold]['train']
        test = self.structure[fold]['test']

        X_train, X_test, y_train, y_test = [], [], [], []

        for elem_id in train:
            x, y = self.read_id(elem_id)
            X_train.append(x)
            if y > 1:
                y = y - 1
            y_train.append(y)

        for elem_id in test:
            x, y = self.read_id(elem_id)
            X_test.append(x)
            if y > 1:
                y = y - 1  
            y_test.append(y)

        return X_train, X_test, y_train, y_test

    def read_id(self, elem_id):
        """
            Return the pair of norms and a class.

            :param elem_id: Encoded id corresponding to a norm pair in dataset.
            :type elem_id: str
        """
        file_origin, index = str(elem_id)[0], int(str(elem_id)[1:])

        if file_origin == ID_CONF:
            df = self.dfs[file_origin]
            row = df[df['conflict_id'] == index]
            if self.binary:
                return (row['norm1'].to_string(index=False),
                row['norm2'].to_string(index=False)), CONF_CLASS
            else:
                return (row['norm1'].to_string(index=False),
                row['norm2'].to_string(index=False)), int(
                row['conf_type'].to_string(index=False))
        elif file_origin == ID_N_CONF:
            df = self.dfs[file_origin]
            row = df[df['index'] == index]
            return (row['norm1'].to_string(index=False),
                row['norm2'].to_string(index=False)), N_CONF_CLASS 
        else:
            print "This is file_origin is wrong: %s" % file_origin


def main():
    # Set Folds class.
    f = Folds(FOLDS_PATH, DATA_PATH[ID_CONF], DATA_PATH[ID_N_CONF], BINARY)
    X_test, y_test = f.read_test()
    X_train, X_val, y_train, y_val = f.read_fold('5')


if __name__ == '__main__':
    main()
