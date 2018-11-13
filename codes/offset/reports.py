#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script contains functions to generate reports
"""
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import filehandler as fh


def accuracy_per_class(cm):
    row, col = cm.shape
    lineVal, colVal, accuracy = [], [], []
    colVal.append(cm.sum(axis=0, dtype='float'))
    lineVal.append(cm.sum(axis=1, dtype='float'))

    for i in range(row):
        accuracy.append(cm[i][i] / (colVal[0][i] + lineVal[0][i] - cm[i][i]))

    print 'Acc. per class:'+ str(accuracy)
    acc_per_class_norm = np.round_(accuracy, decimals=3)
    return acc_per_class_norm


def save_PRF(inputfile, outputfile):
    """
    Create a report containing Precision, Recall and F-measure
    of all classes
    """
    logger.info('Saving file: %s' % outputfile)
    fout = open(outputfile, 'w')

    labels, preds = fh.read_labels_file(inputfile)
    names = map(str, list(set(sorted(labels))))

    fout.write(classification_report(labels, preds, target_names=names))
    fout.write('\n\n')
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(np.array(labels, dtype=float), np.array(preds, dtype=float))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_round = np.round_(cm_normalized, decimals=2)
    acc_norm = accuracy_per_class(cm)
    fout.write(np.array2string(cm))
    fout.write('\n\n')
    fout.write(np.array2string(cm_round))
    fout.write('\n\n')
    fout.write('Accuracy: '+str(acc))
    fout.write('\n\n')
    fout.write('Accuracy Per Class: ')
    fout.write(np.array2string(acc_norm))
    
