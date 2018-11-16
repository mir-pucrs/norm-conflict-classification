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

def create_vectors(cfpairs, dfcf, mode, labels):
    X = embeddings.apply_mode(cfpairs, dfcf, mode=mode)
    X = np.array(X).astype(float)
    y = np.array(labels).astype(int)
    logger.debug('Shape of X matrix: (%d, %d)' % X.shape)
    logger.debug('Shape of y matrix: %d' % y.shape)
    return X, y


def main(fdfile, cffile, ncfile, output):
    #fd=folds :: cf=conflict :: nc=non-conflict
    fdfile = fh.is_file(fdfile)
    cffile = fh.is_file(cffile)
    ncfile = fh.is_file(ncfile)
    cffemb = fh.check_output(None, cffile, prefix='emb')
    output = fh.is_file(output, boolean=True)
    if not output:
        output = join(dirname(cffile), 'svm_conflicts.txt')
    
    results = utils.KfoldResults(binary=False)

    cfemb = fh.DFrameEmbeddings(cffemb)

    # process folds
    fout = open(output, 'w')
    fout.write('# id_1 id_2 ground pred\n')
    fds = fh.Folds(fdfile, cffile, ncfile)
    cf_pairs, cf_labels, _, _ = fds.test()

    logger.info('Train SVM classifier with C=%.1f' % conf.get('SVM_C'))
    for id_fold, arr in enumerate(fds):
        logger.info('Processing FOLD %d' % id_fold)
        # train: tr :: test: ts
        cftr_pairs, cftr_labels, _, _, cfts_pairs, cfts_labels, _, _ = arr
        X_train, y_train = create_vectors(cftr_pairs, cfemb, conf.get('MODE'), cftr_labels)
        # Train SVM classifier
        clf = svm.LinearSVC(multi_class='crammer_singer', C=conf.get('SVM_C'))
        clf.fit(X_train, y_train)
    
        X_test, y_test = create_vectors(cfts_pairs, cfemb, conf.get('MODE'), cfts_labels)
        y_pred = clf.predict(X_test)

        fout.write('# fold: %d\n' % id_fold)        
        for arr, y, p in zip(cfts_pairs, y_test, y_pred):
            id1, id2 = arr
            fout.write('%d %d %d %d\n' % (id1, id2, y, p))
        results.add(id_fold, y_test, y_pred)

    # add the mean of each measure to the results
    results.add_mean()
    
    # get best model and generate the offset
    best_fold, best_score = results.get_best(metric=conf.get('BEST_SCORE'))
    logger.info('Loading best fold: %d (%s: %f)' % (best_fold, conf.get('BEST_SCORE'), best_score))

    arr = fds.get_fold(best_fold)
    cftr_pairs, cftr_labels, _, _, cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = arr
    X_train, y_train = create_vectors(cftr_pairs, cfemb, conf.get('MODE'), cftr_labels)
    # Train SVM classifier
    clf = svm.LinearSVC(multi_class='crammer_singer', C=conf.get('SVM_C'))
    clf.fit(X_train, y_train)

    X_test, y_test = create_vectors(cf_pairs, cfemb, conf.get('MODE'), cf_labels)
    y_pred = clf.predict(X_test)

    fout.write('# fold: test\n')     
    for arr, y, p in zip(cfts_pairs, y_test, y_pred):
        id1, id2 = arr
        fout.write('%d %d %d %d\n' % (id1, id2, y, p))
    results.add('test', y_test, y_pred)
    results.save(join(dirname(cffile), 'results_conflicts_svm.txt'))
    fout.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file_folds', metavar='json_folds', help='File containing train/test folds')
    argparser.add_argument('conflict_file', metavar='conflict_pairs', help='File containing pairs of norms with conflicts')
    argparser.add_argument('nonconflict_file', metavar='non_conflict_pairs', help='File containing pairs of norms without conflicts')
    argparser.add_argument('-o', '--output', help='File to save distances', default=None)
    args = argparser.parse_args()
    main(args.file_folds, args.conflict_file, args.nonconflict_file, args.output)
   
