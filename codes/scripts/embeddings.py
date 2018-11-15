#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script deals with embedding creation from sentences and 
generation of offsets between pairs of norms.
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
import random
random.seed(32)
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import metrics
from os.path import dirname, basename, join, realpath

import filehandler as fh
import parser
import utils
import progressbar

# Define constants
CONFLICT = 1
N_CONFLICT = 0

def embeddings_from_norms(normfile, output, wikiuni, jar, fasttext):
    """
    Generate embeddings from norms 

    Parameters:
    -----------
    normfile: string
        Path to the file containing the following structure
        con_id,conf_id,conf_type,norm1,norm2
    output: string
        Path to a folder to save the embeddings
    wikiuni: string
        Path to the Wikipedia embeddings
    jar: string
        Path to the JAR file of the Stanford tagger
    fasttext: string
        Path to the executable of FastText 
    header: string
        Type of header of the CSV file
    """ 
    # Create output paths to files
    path_emb = fh.check_output(output, normfile, prefix="emb")

    # Read norms from conflicts
    dpairs = fh.NormPairsFile(normfile)
    dnorms = dpairs.norm_dictionary()
    
    logger.info('Creating embeddings for %d sentences' % len(dnorms))

    # Generate embeddings
    embeddings = parser.sentence_embeddings(wikiuni, jar, fasttext, dnorms.keys(), ngram='unigrams', model='wiki')
    
    with open(path_emb, 'w') as femb:
        dims = ['d'+str(i) for i in range(len(embeddings[0]))]
        femb.write('id;'+';'.join(dims)+'\n')
        pb = progressbar.ProgressBar(len(dnorms.keys()))
        for sent, e in zip(dnorms, embeddings):
            id = dnorms[sent]
            str_emb = map(str, e)
            str_emb = u';'.join(str_emb)
            femb.write('%d;%s\n' % (id, str_emb))
            pb.update()


def generate_offset(pairs, dfemb):
    """
    From a set of pairs of norms and their embeddings, 
    create an offset vector.

    Parameters:
    -----------
    pairs: list of tuples (int)
        [(id1, id2), (id3, id4)...]
    dfemb: pandas.dataframe
        Dataframe containing embeddings of norms in the form:
        id;dim0;dim1;...;dim599
    """
    emb_sum = np.zeros(dfemb.size_embedding())
    for id1, id2 in pairs:
        emb1 = dfemb.id2embed(id1)
        emb2 = dfemb.id2embed(id2)
        if len(emb1) != len(emb2):
            logger.error('Embeddings with different sizes (%d:%d)' % (len(emb1), len(emb2)))
            logger.error(emb1)
            logger.error(emb1)
            sys.exit(0)
        emb_sum += emb1 - emb2
    offset = emb_sum / len(pairs)
    logger.debug('Generated offset from %d pairs.' % len(pairs))
    return offset


def generate_offset_from_file(normfile, embfile, output):
    """ 
    Create offset vector from embedding vectors 

    Parameters:
    -----------
    normfile: string
        Path to the file containing pairs of norms
    embfile: string
        Path to the file containing embeddings of norms
    output: string
        Path to the file to save the output vector
    """
    dnorms = fh.NormPairsFile(normfile)
    pairs = dnorms.id_pairs()
    dfemb = fh.DFrameEmbeddings(embfile)
    offset = generate_offset(pairs, dfemb)
    np.savetxt(output, offset)
    return offset


def calculate_distance_offset(pairs, df, offset, offset2=None):
    """
    From a set of pairs, calculate the distance of the offset of the 
    pair in relation to the global offset.

    Parameters:
    -----------
    pairs: list
        list containing tuples of IDs of norms [(id1,id2),(id3,id4)...]
    df: pandas.dataframe
        dataframe containing ids and embeddings of sentences
    offset: np.array
        vector containing the global offset (offset of all conflicts)
    """
    label = 0
    vdist = []
    pb = progressbar.ProgressBar(len(pairs))
    for i, arr in enumerate(pairs):
        emb1 = df.id2embed(arr[0])
        emb2 = df.id2embed(arr[1])
        local_offset = emb1 - emb2

        # cosine (similar:0->2:not_similar)
        cos = utils.cosine(local_offset, offset)
        # euclidean distance (similar:0->inf:not_similar)
        euc = utils.euclidean(local_offset, offset)
        if len(offset2) > 0:
            cos2 = utils.cosine(local_offset, offset2)
            euc2 = utils.euclidean(local_offset, offset2)
            vdist.append([cos, euc, cos2, euc2])
        else:
            vdist.append((cos, euc))
        pb.update()
        #if i == 1000: break
    return vdist


def related_offset(pairs, df, offset, n, measure='cos', metric='closest', dist=False):
    """
    Get pairs that are somehow related to the offset.

    Parameters:
    -----------
    pairs: list
        List containing tuples of IDs of norms [(id1,id2),(id3,id4)...]
    df: pandas.dataframe
        Dataframe containing ids and embeddings of sentences
    offset: np.array
        Vector containing the global offset (offset of all conflicts)
    n: int
        Number of elements to return
    measure: string (cos|euc)
        Select between cosine and euclidean distance
    metric: string (closest|farthest|random)
        Metric to select the non-conflict pairs
    dist: boolean
        Return the distance in the output array
    """
    if metric == 'random':
        vec = random.sample(pairs, n)
        return vec

    v_dist = calculate_distance_offset(pairs, df, offset)
    v = []
    for i in range(len(v_dist)):
        #ds = c, e
        if measure == 'cos':
            v.append([v_dist[i][0]]+list(pairs[i]))
        elif measure == 'euc':
            v.append([v_dist[i][1]]+list(pairs[i]))
        else:
            logger.erro('Measure %s not applicable' % measure)
            sys.exit(0)

    if metric == 'closest':
        vred = sorted(v)[:n]
    elif metric == 'farthest':
        vred = sorted(v, reverse=True)[:n]
    elif metric == 'random':
        vred = random.sample(v, n)

    vec = []
    for arr in vred:
        if dist:
            inv = arr[1:]+arr[0]
            vec.append(tuple(inv))
        else:
            vec.append(arr[1:])
    return vec
