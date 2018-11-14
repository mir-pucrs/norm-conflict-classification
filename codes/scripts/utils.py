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
