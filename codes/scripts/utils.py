#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script contains util functions
"""
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
from os.path import realpath
from scipy.spatial import distance
from sklearn import metrics


def combine(pair, df, mode='offset'):
    """
    Generates the training vectors from the input file

    Parameters:
    -----------
    pairs_class: list
        List containing tuples of pairs of norms and their class
    df: pandas.dataframe
        Dataframe containing embeddings
    """
    emb1 = df.id2embed(pair[0])
    emb2 = df.id2embed(pair[1])
    if mode == 'offset':
        comb = emb1 - emb2
    elif mode == 'concat':
        comb = np.concatenate((emb1,emb2))
    elif mode == 'mean':
        comb = (emb1 + emb2)/2.
    return comb


def load_offset(offset_file):
    """ Load the content of the offset file """
    offset_file = realpath(offset_file)
    return np.loadtxt(offset_file)


def cosine(v1, v2):
    """ Calculate the cosine distance """
    cos = distance.cosine(v1, v2)
    return float(cos)


def euclidean(v1, v2):
    """ Calculate the Euclidean distance """
    euc = metrics.euclidean_distances([v1], [v2])
    return float(euc[0][0])


def calculate_distance(v1, v2, measure='euc'):
    """ Calculate distances between two vectors """
    if not isinstance(v1, np.ndarray): 
        v1 = np.array(v1)
    if not isinstance(v2, np.ndarray): 
        v1 = np.array(v2)
    if measure == 'euc':
        return euclidean(v1, v2)
    elif measure == 'cos':
        return cosine(v1, v2)
    else:
        logger.error('Measure %s not implemented!' % measure)
        sys.error(0)


class KfoldResults(object):
    def __init__(self, binary=True):
        self.max_acc = 0
        self.dresults = {}
        self.binary = binary


    def add(self, nb_fold, ground, predicted):
        acc = metrics.accuracy_score(ground, predicted)
        if self.binary:
            prec, rec, fscore, _ = metrics.precision_recall_fscore_support(ground, predicted, average='binary')
        else:
            prec, rec, fscore, _ = metrics.precision_recall_fscore_support(ground, predicted)
        self.dresults[nb_fold] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f-score': fscore
        }


    def get_best(self, metric='accuracy'):
        best_val = 0
        best_fold = 0
        for nb_fold in self.dresults:
            if self.dresults[nb_fold].has_key(metric):
                score = self.dresults[nb_fold][metric]
                if score > best_val:
                    best_val = score
                    best_fold = nb_fold
            else:
                logger.error('Metric %s is not implemented' % metric)
                sys.exit(0)
        return best_fold, best_val


    def add_mean(self):
        """ Return the mean of each measure """
        acc, prec, rec, fscore = 0, 0, 0, 0
        for nb_fold in self.dresults:
            acc += self.dresults[nb_fold]['accuracy']
            prec += self.dresults[nb_fold]['precision']
            rec += self.dresults[nb_fold]['recall']
            fscore += self.dresults[nb_fold]['f-score']
        mean_acc = float(acc)/len(self.dresults)
        mean_prec = float(prec)/len(self.dresults)
        mean_rec = float(rec)/len(self.dresults)
        mean_fscore = float(fscore)/len(self.dresults)
        self.dresults['mean'] = {
            'accuracy': mean_acc,
            'precision': mean_prec,
            'recall': mean_rec,
            'f-score': mean_fscore
        }
        return mean_acc, mean_prec, mean_rec, mean_fscore


    def save(self, fname):
        """ Save results in a file """
        with open(fname, 'w') as fout:
            for nb_fold in self.dresults:
                fout.write('#Fold: %s\n' % str(nb_fold))
                fout.write('- Accuracy: %f\n' % self.dresults[nb_fold]['accuracy'])
                fout.write('- Precision: %f\n' % self.dresults[nb_fold]['precision'])
                fout.write('- Recall: %f\n' % self.dresults[nb_fold]['recall'])
                fout.write('- F-score: %f\n' % self.dresults[nb_fold]['f-score'])
