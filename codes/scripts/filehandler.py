#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script contains util functions
"""
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from os.path import dirname, basename, join, realpath, isfile, splitext
from collections import OrderedDict
import numpy as np
import json
import pandas as pd
pd.options.display.max_colwidth = 1000

import fileconfig
import progressbar

# Set constants
CONFLICT = '2'
NN_CONFLICT = '1'

def is_file(inputfile, boolean=False):
    """ Check whether the ``inputfile`` corresponds to a file """
    if not inputfile or not isfile(inputfile):
        if boolean:
            return False
        logger.error('Input is not a file!')
        sys.exit(0)
    inputfile = realpath(inputfile)
    return inputfile


def is_folder(inputfolder, boolean=False):
    """ Check whether the ``inputfolder`` corresponds to a folder """
    if not inputfolder or not isdir(inputfolder):
        if boolean:
            return False
        logger.error('Argument %s is not a folder!' % inputfolder)
        sys.exit(0)
    inputfolder = realpath(inputfolder)
    return inputfolder


def check_output(output, input, prefix="emb", suffix="csv"):
    """ Return a valid output """
    fileinput = realpath(input)
    dirin = dirname(input)
    fname, ext = splitext(basename(input))
    if not output:
        output = join(dirin, prefix+'_'+fname+'.'+suffix)
    else:
        output = join(output, prefix+'_'+fname+'.'+suffix)
    output = realpath(output)
    return output


def read_labels_file(inputfile):
    """ Read a text file containing true and predicted labels """
    labels, preds = [], []
    with open(inputfile) as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith('#'):
                y, p = line.split()
                labels.append(int(y))
                preds.append(int(p))
    return labels, preds


class NormPairsFile(object):
    """
    File containing conflicts or non-conflicts in the form
        con_id,conf_id,conf_type,norm1,norm2
    """
    def __init__(self, input):
        self.fc = fileconfig.Configuration()
        self.input = input
        self.df = pd.read_csv(input)
        self.nb_rows = len(self.df)
        self.pairs = []
        logger.info('Loading %d norms' % self.nb_rows)
        if not 'sent_id_1' in self.df.columns:
            self._create_id_columns()


    def __iter__(self):
        pb = progressbar.ProgressBar(self.nb_rows)
        for i in range(self.nb_rows):
            id1 = self.df['sent_id_1'][i]
            norm1 = self.df['norm1'][i]
            id2 = self.df['sent_id_2'][i]
            norm2 = self.df['norm2'][i]
            pb.update()
            yield id1, norm1, id2, norm2


    def _create_id_columns(self):
        """
        Create ids for sentences in order to have each sentence
        with a unique id, thus generating a single embedding for
        sentence.
        """
        id = 0
        dsent = {}
        id1, id2 = [], []
        for i in range(self.nb_rows):
            norm1 = self.df['norm1'][i]
            if not dsent.has_key(norm1):
                dsent[norm1] = id
                id += 1
            id1.append(dsent[norm1])

            norm2 = self.df['norm2'][i]
            if not dsent.has_key(norm2):
                dsent[norm2] = id
                id += 1
            id2.append(dsent[norm2])

        self.df.insert(loc=0, column='sent_id_1', value=id1)
        self.df.insert(loc=1, column='sent_id_2', value=id2)
        self.df.to_csv(self.input, index=False)
        self.df = pd.read_csv(self.input)
        

    def id_norm_pairs(self):
        for id1, norm1, id2, norm2 in self:
            self.pairs.append((id1, norm1, id2, norm2))
        return self.pairs


    def id_pairs(self):
        for id1, _, id2, _ in self:
            self.pairs.append((id1, id2))
        return self.pairs


    def pairs_class(self):
        for i in range(self.nb_rows):
            id1 = self.df['sent_id_1'][i]
            id2 = self.df['sent_id_2'][i]
            label = self.df['conf_type'][i]
            self.pairs.append((id1, id2, label))
        return self.pairs

    
    def norm_dictionary(self):
        dic = OrderedDict()
        for id1, norm1, id2, norm2 in self:
            dic[norm1] = id1
            dic[norm2] = id2
        return dic


    def pair_from_index(self, index):
        """ Return the norm pair corresponding to the index """
        if 'conflict_id' in self.df:
            row = self.df[self.df['conflict_id'] == index]
            return (row['sent_id_1'].values[0], 
                    row['sent_id_2'].values[0], 
                    row['conf_type'].values[0])
        elif 'index' in self.df:
            row = self.df[self.df['index'] == index]
            return (row['sent_id_1'].values[0], 
                    row['sent_id_2'].values[0], 
                    self.fc.get('N_CONFLICT'))
        else:
            logger.error('CSV file does not contain a column with identifier.')
            sys.exit(0)
#End of class NormPairsFile


class DFrameEmbeddings(object):
    def __init__(self, csvfile):
        self.df = pd.read_csv(csvfile, sep=';')
        self.df[['id']] = self.df[['id']].applymap(np.int64)


    def head(self):
        return self.df.head()


    def size_embedding(self):
        return len(self.df.iloc[0,1:])


    def id2embed(self, id):
        dfrow = self.df.loc[self.df['id'] == id]
        try:
            row = dfrow.values.tolist()[0][1:]
        except:
            logger.warning(dfrow)
            logger.warning(id)
        return np.array(row)
#End of class DFrameEmbeddings


class Folds(object):
    def __init__(self, json_file, cffile, ncfile):
        """ 
        json_file: string
            path to the JSON file containing k-fold splitting
        cffile: string
            path to the CSV file containing conflicting pairs
        ncfile: string
            path to the CSV file containing non-conflicting pairs
        """
        self.dfolds = json.load(open(json_file))
        self.nb_folds = len(self.dfolds.keys()) - 1
        self.cfdata = NormPairsFile(cffile)
        self.ncdata = NormPairsFile(ncfile)


    def __iter__(self):
        """
        Iterate on all folds from json file and yield
        conflict and non-conflict ids from train and test
        """
        for id_fold in range(self.nb_folds):
            arr = self.dfolds[str(id_fold)]
            vtrain, vtest = arr['train'], arr['test']

            arr = self._indexes2ids(vtrain)
            cftr_pairs, cftr_labels, nctr_pairs, nctr_labels = arr
            arr = self._indexes2ids(vtest)
            cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = arr
            yield (cftr_pairs, cftr_labels, nctr_pairs, nctr_labels,
                   cfts_pairs, cfts_labels, ncts_pairs, ncts_labels)


    def _indexes2ids(self, vfold):
        cf_pairs, cf_labels = [], []
        nc_pairs, nc_labels = [], []
        for id in vfold:
            type, index = str(id)[0], int(str(id)[1:])
            if type == CONFLICT:
                id1, id2, label = self.cfdata.pair_from_index(index)
                cf_pairs.append((id1, id2))
                cf_labels.append(label)
            elif type == NN_CONFLICT:
                id1, id2, label = self.ncdata.pair_from_index(index)
                nc_pairs.append((id1, id2))
                nc_labels.append(label)
            else:
                logger.error('Unknown conflict with ID: %s' % id)
                logger.error('Norms must start with 1 for non-conflict and 0 for conflict')
                sys.exit(0)
        return (cf_pairs, cf_labels, nc_pairs, nc_labels)


    def get_fold(self, nb_fold):
        arr = self.dfolds[str(nb_fold)]
        vtrain, vtest = arr['train'], arr['test']
        cftr_pairs, cftr_labels, nctr_pairs, nctr_labels = self._indexes2ids(vtrain)
        cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = self._indexes2ids(vtest)
        return (cftr_pairs, cftr_labels, nctr_pairs, nctr_labels,
                cfts_pairs, cfts_labels, ncts_pairs, ncts_labels)


    def test(self):
        """ Get the test set from json file """
        arr = self._indexes2ids(self.dfolds['test'])
        cf_pairs, cf_labels, nc_pairs, nc_labels = arr
        return (cf_pairs, cf_labels, nc_pairs, nc_labels)
#End of class Folds


def embeddings_from_indexes(normfile, embfile, indexes):
    """
    Return a matrix of embeddings and true labels from a list of indexes
    representing unique norm pairs
    """
    vemb, y = [], []
    fnorm = NormPairsFile(normfile)
    femb = DFrameEmbeddings(embfile)
    
    for index in indexes:
        id1, id2, lbl = fnorm.pair_from_index(index)
        emb1 = femb.id2embed(id1)
        emb2 = femb.id2embed(id2)
        vemb.append([emb1, emb2])
        y.append(lbl)
    return vemb, y


def exclude_existing_pairs(normpairs, existing_pairs):
    """ 
    Return a list of pairs excluding the existing pairs 

    filenorms: string
        path to a CSV file containing non-conflicting pairs
    existing_pairs: array
        list of pair ids
    """
    all_pairs = normpairs.id_pairs()
    pairs = set(all_pairs) - set(existing_pairs)
    return list(pairs)
