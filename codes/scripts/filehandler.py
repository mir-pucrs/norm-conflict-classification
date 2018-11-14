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
import pandas as pd
from collections import OrderedDict
import numpy as np

import progressbar

def is_file(inputfile, boolean=False):
    """ Check whether the ``inputfile`` corresponds to a file """
    inputfile = realpath(inputfile)
    if not isfile(inputfile):
        if boolean:
            return False
        logger.error('Input is not a file!')
        sys.exit(0)
    return inputfile


def is_folder(inputfolder, boolean=False):
    """ Check whether the ``inputfolder`` corresponds to a folder """
    inputfolder = realpath(inputfolder)
    if not isdir(inputfolder):
        if boolean:
            return False
        logger.error('Argument %s is not a folder!' % inputfolder)
        sys.exit(0)
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
            print dfrow
            print '>', id
        return np.array(row)
#End of class DFrameEmbeddings


