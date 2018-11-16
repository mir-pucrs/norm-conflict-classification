#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Train an SVM on norm conflicts data and test the generated model.
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from os.path import join, dirname
import numpy as np
from sklearn import svm

from scripts import filehandler as fh
from scripts import embeddings
from scripts import fileconfig as fc
from scripts import utils

conf = fc.Configuration()

def create_vectors(cfpairs, dfcf, ncpairs, dfnc, mode, binary, labels=None):
    X_train = embeddings.apply_mode(cfpairs, dfcf, mode=mode)
    if binary:
        y_train = [conf.get('CONFLICT') for i in range(len(X_train))]
    else:
        y_train = labels
    X_temp = embeddings.apply_mode(ncpairs, dfnc, mode=mode)
    y_temp = [conf.get('N_CONFLICT') for i in range(len(X_temp))]
    X_train.extend(X_temp)
    y_train.extend(y_temp)
    X_train = np.array(X_train).astype(float)
    y_train = np.array(y_train).astype(int)
    logger.debug('Shape of X matrix: (%d, %d)' % X_train.shape)
    logger.debug('Shape of y matrix: %d' % y_train.shape)
    return X_train, y_train


def main(fdfile, cffile, ncfile, output):
    #fd=folds :: cf=conflict :: nc=non-conflict
    fdfile = fh.is_file(fdfile)
    cffile = fh.is_file(cffile)
    ncfile = fh.is_file(ncfile)
    cffemb = fh.check_output(None, cffile, prefix='emb')
    ncfemb = fh.check_output(None, ncfile, prefix='emb')
    output = fh.is_file(output, boolean=True)
    if not output:
        output = join(dirname(cffile), 'svm.txt')
    
    results = utils.KfoldResults(binary=conf.get('BINARY'))

    cfemb = fh.DFrameEmbeddings(cffemb)
    ncemb = fh.DFrameEmbeddings(ncfemb)

    # process folds
    fout = open(output, 'w')
    fout.write('# id_1 id_2 ground pred\n')
    fds = fh.Folds(fdfile, cffile, ncfile)
    cf_pairs, cf_labels, nc_pairs, nc_labels = fds.test()

    logger.info('Train SVM classifier with C=%.1f' % conf.get('SVM_C'))
    for id_fold, arr in enumerate(fds):
        logger.info('Processing FOLD %d' % id_fold)
        # train: tr :: test: ts
        cftr_pairs, cftr_labels, nctr_pairs, nctr_labels, cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = arr
        X_train, y_train = create_vectors(cftr_pairs, cfemb, nctr_pairs, ncemb, conf.get('MODE'), conf.get('BINARY'), labels=cftr_labels)
        # Train SVM classifier
        clf = svm.LinearSVC(multi_class='crammer_singer', C=conf.get('SVM_C'))
        clf.fit(X_train, y_train)
    
        X_test, y_test = create_vectors(cfts_pairs, cfemb, ncts_pairs, ncemb, conf.get('MODE'), conf.get('BINARY'), labels=cfts_labels)
        y_pred = clf.predict(X_test)
        X_ids = cfts_pairs + ncts_pairs

        fout.write('# fold: %d\n' % id_fold)        
        for arr, y, p in zip(X_ids, y_test, y_pred):
            id1, id2 = arr
            fout.write('%d %d %d %d\n' % (id1, id2, y, p))
        results.add(id_fold, y_test, y_pred)

    # add the mean of each measure to the results
    results.add_mean()
    
    # get best model and generate the offset
    best_fold, best_score = results.get_best(metric=conf.get('BEST_SCORE'))
    logger.info('Loading best fold: %d (%s: %f)' % (best_fold, conf.get('BEST_SCORE'), best_score))

    arr = fds.get_fold(best_fold)
    cftr_pairs, cftr_labels, nctr_pairs, nctr_labels, cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = arr
    X_train, y_train = create_vectors(cftr_pairs, cfemb, nctr_pairs, ncemb, conf.get('MODE'), conf.get('BINARY'), labels=cftr_labels)
    # Train SVM classifier
    clf = svm.LinearSVC(multi_class='crammer_singer', C=conf.get('SVM_C'))
    clf.fit(X_train, y_train)

    X_test, y_test = create_vectors(cf_pairs, cfemb, nc_pairs, ncemb, conf.get('MODE'), conf.get('BINARY'), labels=cf_labels)
    y_pred = clf.predict(X_test)
    X_ids = cf_pairs + nc_pairs

    fout.write('# fold: test\n')     
    for arr, y, p in zip(X_ids, y_test, y_pred):
        id1, id2 = arr
        fout.write('%d %d %d %d\n' % (id1, id2, y, p))
    results.add('test', y_test, y_pred)
    results.save(join(dirname(cffile), 'results_svm.txt'))
    fout.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file_folds', metavar='json_folds', help='File containing train/test folds')
    argparser.add_argument('conflict_file', metavar='conflict_pairs', help='File containing pairs of norms with conflicts')
    argparser.add_argument('nonconflict_file', metavar='non_conflict_pairs', help='File containing pairs of norms without conflicts')
    argparser.add_argument('-o', '--output', help='File to save distances', default=None)
    args = argparser.parse_args()
    main(args.file_folds, args.conflict_file, args.nonconflict_file, args.output)
   
