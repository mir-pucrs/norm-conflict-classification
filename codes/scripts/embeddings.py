#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script will take all the images from inputfile.txt and classify according the neural net.
It will create an output file that have the path from the image, the correct label, and the predict label.
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


def calculate_distance_offset(pairs, df, offset):#
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
        vdist.append((cos, euc))
        pb.update()
        #if i == 1000: break
    return vdist


def related_offset(pairs, df, offset, n, measure='cos', metric='closest', dist=False):
    """
    Get pairst that are somehow related to the offset.

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
########


def generate_embeddings_from_norms(norm_file, output=None):
    """
    From a set of sentences contained in `norm_file`, generates a CSV file 
    containing an id, the sentence and its embedding values separated
    by semicolon.
    """ 
    norm_file = realpath(norm_file)
    output = fh.check_output(output, norm_file, prefix="emb")

    # Read norms from conflicts
    csv = fh.CsvNorms(norm_file)
    seen_norms = csv.norm_dictionary().keys()

    # Generating embeddings
    dids = {}
    embeddings = parser.get_sentence_embeddings(seen_norms, ngram='unigrams', model='wiki')
    with open(output, 'w') as fp:
        dims = ['d'+str(i) for i in range(len(embeddings[0]))]
        fp.write('id;sentence;'+';'.join(dims)+'\n')
        pb = progressbar.ProgressBar(len(seen_norms))
        id = 0
        for s, e in zip(seen_norms, embeddings):
            s = s.encode('utf-8')
            s = s.replace(';', '.,')
            dids[s] = id
            str_e = map(str, e)
            str_e = u';'.join(str_e)
            fp.write('%d;%s;%s\n' % (id, s, str_e))
            id += 1
            pb.update()

    # Save file of pairs containing id instead of senteces
    csv = fh.CsvNorms(norm_file)
    norm_pairs = csv.norm_pairs()
    outids = fh.check_output(None, norm_file, prefix="ids")
    with open(outids, 'w') as fout:
        fout.write('norm1,norm2\n')
        for norm1, norm2 in norm_pairs:
            fout.write('%d,%d\n' % (dids[norm1], dids[norm2]))
    


def calculate_distances(emb_file, output=None):
    """ 
    Generate a CSV file containing the id of each embedding along with
    the cosine and euclidean distance between embeddings.
    """
    output = fh.check_output(output, emb_file, prefix="dist")
    emb_file = realpath(emb_file)
    content = fh.load_embeddings(emb_file)

    fout = open(output, 'w')
    logger.info('measuring distance of vectors')
    pb = progressbar.ProgressBar(len(content))
    for i in range(len(content)):
        for j in range(i+1, len(content)):
            id1, emb1 = content[i]
            id2, emb2 = content[j]
            cosdist = utils.cosine(emb1, emb2)
            cosdist = float(cosdist)
            eucdist = utils.euclidean_distances([emb1], [emb2])
            eucdist = float(eucdist[0][0])
            fout.write('%s;%s;%f;%f\n' % (id1, id2, cosdist, eucdist))
        pb.update()
    fout.close()









def dn_closest_offset(pairs, df, n, offset, measure='cos'):
    """
    Get the closest pairst to the offset

    Parameters:
    -----------
    pairs: list
        list containing tuples of IDs of norms [(id1,id2),(id3,id4)...]
    df: pandas.dataframe
        dataframe containing ids and embeddings of sentences
    n: int
        number of conflicts in dataset
    offset: np.array
        vector containing the global offset (offset of all conflicts)
    measure: string (cos|euc)
        select between cosine and euclidean distance
    """
    v_dist = calculate_distance_offset(pairs, df, offset)
    
    # transform list into a Dataframe
    idcols = ['norm1','norm2']
    df_dist = pd.DataFrame(v_dist, columns=['norm1','norm2','cos','euc'])
    df_dist[idcols] = df_dist[idcols].applymap(np.int64)
    df_dist = df_dist.sort_values(by=[measure])
    df_dist = df_dist.head(n)
    return df_dist.ix[:, ['norm1', 'norm2', measure]]

    
def distance_offset_from_file(file_pairs, file_emb, file_offset, output):#
    """
    Generate a file containing ids of norms and their distance 
    to the global offset using cosine and euclidean distances.
    
    Parameters:
    -----------
    fids_noncft: string
        path to the file containing pairs of ids for non-conflicting norms
    emb_noncft: string
        path to the file containing embeddings corresponding to ids of norms
    foffset: string 
        path to the file containing the global offset (saved with numpy)
    output: string
        path to the CSV output file to save pairs of ids, cosine and euclidean distances
    """
    offset = utils.load_offset(file_offset)
    df_pairs = fh.DFrameCSV(file_pairs)
    vpairs = df_pairs.pairs()
    df_emb = fh.DFrameEmbeddings(file_emb)
    vdist = calculate_distance_offset(vpairs, df_emb, offset)
    with open(output, 'w') as fout:
        for n1, n2, c, e in vdist:
            fout.write('%d,%d,%f,%f\n' % (n1, n2, c, e))





def test_conflict(df_emb, pair, offset, offset_non, measure):
    """
    From a pair of norms and their embeddings, compare the offset
    distance of both norms with the offset of conflicts and non-conflicts
    
    Parameters:
    -----------
    df_emb: pandas.dataframe
        Dataframe containing embeddings for the pair of norms
    pair: tuple
        Norm1 and norm2
    offset: np.array
        Vector containing the offset of conflicting norms
    offset_non: np.array
        Vector containing the offset of non-conflicting norms
    measure: string (euc|cos)
        Euclidean or Cosine functions to measure the distance
    """
    n1, n2 = pair
    emb1 = df_emb.id2embed(n1)
    emb2 = df_emb.id2embed(n2)
    diff = emb1 - emb2

    if measure == 'euc':
        dist_cft = utils.euclidean(diff, offset) 
        dist_non = utils.euclidean(diff, offset_non)
    elif measure == 'cos':
        dist_cft = utils.cosine(diff, offset) 
        dist_non = utils.cosine(diff, offset_non)
    else:
        logger.error('Measure %s not implemented (euc|cos)!' % measure)
        sys.exit(0)

    if dist_cft < dist_non:
        return CONFLICT
    elif dist_cft > dist_non:
        return N_CONFLICT
    else:
        logger.warning('Could not define whether it is a conflict or not! Setting NO conflict')
    return N_CONFLICT





def load_embeddings(file_embeddings):
    df_csv = pd.read_csv(file_embeddings, sep=';')
    return df_csv
