import os
import csv
import numpy as np
from itertools import combinations
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from constants import *
from norm_identifier.sentence_classifier import SentenceClassifier


def get_norms():
    # Get norms from contracts for manually checking.

    folders = os.listdir(BASE_FOLDER_PATH)
    sent_cls = SentenceClassifier()
    sent_cls.load_classifier(CLASSIFIER_PATH)
    names = [n[:-1] for n in open(NAMES_PATH, 'r').readlines()]
    vec = CountVectorizer(vocabulary=names)

    for folder in folders:
        # Run over folders with created norms.
        files = os.listdir(BASE_FOLDER_PATH + folder)

        for file in files:
            # These folders usually have two files, one with the contract and another with the inserted norms.
            # Here we get identify the norms in the contract to manually see which ones were modified.
            if 'new_norms' in file:
                created_norms = open(BASE_FOLDER_PATH + folder + "/" + file, 'r').readlines()
            else:
                contract_text = open(BASE_FOLDER_PATH + folder + "/" + file, 'r').read()
                sentences = sent_tokenize(contract_text)

        # transform = sentences
        norms = []

        for sent in sentences:
            # Run over the sentences to find those that are norms.

            # Transform the sentence into vector of numbers.
            t = vec.fit_transform([sent])
            clss = sent_cls.classifier.predict(t.A)

            if clss[0] == 1:
                norms.append(sent)

        with open(BASE_FOLDER_PATH + folder + '/norms.txt', 'w') as w_file:

            for norm in norms:
                w_file.write(norm + '\n')

        w_file.close()

def create_conflict_free_set():

    norm_file_name = "norms.txt"
    output_std_name = "conflict_free.txt"

    folders = os.listdir(BASE_FOLDER_PATH)

    for folder in folders:
        # Run over folders with created norms.

        print folder
        files = os.listdir(BASE_FOLDER_PATH + folder)
        base_path = BASE_FOLDER_PATH + folder + "/"

        for file in files:

            if "new_norms" in file:
                n_inserted_norms = len(open(base_path + file, 'r').readlines())
                break

        norms = []

        with open(base_path + norm_file_name, 'r') as r_file:

            for norm in r_file.readlines()[:-n_inserted_norms]:

                norms.append(norm.strip())

        output_file = open(base_path + output_std_name, 'w')

        # Create pair of norms.
        # for ind, norm in enumerate(norms):

        #     for index, norm_2 in enumerate(norms):

        #         if ind < len(norms) and ind + index < len(norms):
        #             output_file.write(norms[ind] + '\n' + norms[ind + index] + '\n')
        comb = combinations(norms, 2)

        for c in comb:
            output_file.write(c[0] + '\n' + c[1] + '\n')

def create_datasets():

    folders = os.listdir(BASE_FOLDER_PATH)

    conflict = BASE_FOLDER_PATH + "conflicts.csv"
    non_conflict = BASE_FOLDER_PATH + "non-conflicts.csv"

    for folder in folders:
        # Run over folders with created norms.

        print folder

        conf_lines = open(BASE_FOLDER_PATH + folder + "/conflict.csv").readlines()

        with open(conflict, 'a') as csvfile:
            wrtr = csv.writer(csvfile, delimiter=',')

            for index in range(0, len(conf_lines), 2):
                if index + 1 < len(conf_lines):
                    wrtr.writerow([conf_lines[index].strip(), conf_lines[index+1].strip()])

        non_conf_lines = open(BASE_FOLDER_PATH + folder + "/conflict_free.txt").readlines()

        with open(non_conflict, 'a') as csvfile:
            wrtr = csv.writer(csvfile, delimiter=',')

            for index in range(0, len(non_conf_lines), 2):

                if index + 1 < len(non_conf_lines):
                    wrtr.writerow([non_conf_lines[index].strip(), non_conf_lines[index+1].strip()])

def turn_into_matrix(pairs, maxlength, nb_classes):

    while True:

        for pair in pairs:

            X, y = pair

            matrix_connection = np.zeros(shape=(maxlength, maxlength), dtype=int, order='C')
            norm1, norm2 = X

            for i in xrange(maxlength):

                if i == len(norm1):
                    break

                for j in xrange(maxlength):

                    if j == len(norm2):
                        break

                    if norm1[i] == norm2[j]:
                        matrix_connection[i][j] = 1

            matrix_connection = matrix_connection.reshape(1, maxlength, maxlength, 1)

            yield matrix_connection, y.reshape(1, nb_classes)

if __name__ == "__main__":

    for i in turn_into_matrix([('Hoje tem festa, que tal?', "Horacio e legal, que tal?")], 10):
        print i
