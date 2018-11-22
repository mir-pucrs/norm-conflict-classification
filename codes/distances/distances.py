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

def distance_to_clusters(vdist):
    y_pred = []
    for dist, dist2, id1, id2 in vdist:
        if dist < dist2:
            y_pred.append(conf.get('CONFLICT'))
        else: 
            y_pred.append(conf.get('N_CONFLICT'))
    return y_pred


def main(fdfile, cffile, ncfile, output):
    #fd=folds :: cf=conflict :: nc=non-conflict
    fdfile = fh.is_file(fdfile)
    cffile = fh.is_file(cffile)
    ncfile = fh.is_file(ncfile)
    cffemb = fh.check_output(None, cffile, prefix='emb')
    ncfemb = fh.check_output(None, ncfile, prefix='emb')
    output = fh.is_file(output, boolean=True)
    if not output:
        output = join(dirname(cffile), 'distances.txt')
    
    results = utils.KfoldResults(binary=conf.get('BINARY'))

    cfemb = fh.DFrameEmbeddings(cffemb)
    ncemb = fh.DFrameEmbeddings(ncfemb)

    # process folds
    fout = open(output, 'w')
    fout.write('# id_1 id_2 ground pred\n')
    fds = fh.Folds(fdfile, cffile, ncfile)
    cf_pairs, cf_labels, nc_pairs, nc_labels = fds.test()

    logger.info('Generate clusters by MODE: %s' % conf.get('MODE'))
    for id_fold, arr in enumerate(fds):
        logger.info('Processing FOLD %d' % id_fold)
        # train: tr :: test: ts
        cftr_pairs, cftr_labels, nctr_pairs, nctr_labels, cfts_pairs, cfts_labels, ncts_pairs, ncts_labels = arr

        # compute offset for conflicts and non-conflicts
        cfcluster = embeddings.apply_mode(cftr_pairs, cfemb, mode=conf.get('MODE'), average=True)
        nccluster = embeddings.apply_mode(nctr_pairs, ncemb, mode=conf.get('MODE'), average=True)       
    
        X_ids = cfts_pairs + ncts_pairs
        y_cf = embeddings.calculate_distance_mode(cfts_pairs, cfemb, cfcluster, conf.get('MODE'), conf.get('DISTANCE'), ref2=nccluster)
        y_test = [1]*len(cfts_pairs)
        y_pred = distance_to_clusters(y_cf)
        y_nc = embeddings.calculate_distance_mode(ncts_pairs, ncemb, cfcluster, conf.get('MODE'), conf.get('DISTANCE'), ref2=nccluster)
        y_test += [0]*len(ncts_pairs)
        y_pred += distance_to_clusters(y_nc)

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

    # compute offset for conflicts and non-conflicts
    cfcluster = embeddings.apply_mode(cftr_pairs, cfemb, mode=conf.get('MODE'), average=True)
    nccluster = embeddings.apply_mode(nctr_pairs, ncemb, mode=conf.get('MODE'), average=True)       

    X_ids = cf_pairs + nc_pairs
    y_cf = embeddings.calculate_distance_mode(cf_pairs, cfemb, cfcluster, conf.get('MODE'), conf.get('DISTANCE'), ref2=nccluster)
    y_test = [1]*len(cf_pairs)
    y_pred = distance_to_clusters(y_cf)
    y_nc = embeddings.calculate_distance_mode(nc_pairs, ncemb, cfcluster, conf.get('MODE'), conf.get('DISTANCE'), ref2=nccluster)
    y_test += [0]*len(nc_pairs)
    y_pred += distance_to_clusters(y_nc)

    fout.write('# fold: test\n')        
    for arr, y, p in zip(X_ids, y_test, y_pred):
        id1, id2 = arr
        fout.write('%d %d %d %d\n' % (id1, id2, y, p))
    results.add('test', y_test, y_pred)
    results.save(join(dirname(cffile), 'results_distance.txt'))
    fout.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file_folds', metavar='json_folds', help='File containing train/test folds')
    argparser.add_argument('conflict_file', metavar='conflict_pairs', help='File containing pairs of norms with conflicts')
    argparser.add_argument('nonconflict_file', metavar='non_conflict_pairs', help='File containing pairs of norms without conflicts')
    argparser.add_argument('-o', '--output', help='File to save distances', default=None)
    args = argparser.parse_args()
    main(args.file_folds, args.conflict_file, args.nonconflict_file, args.output)
   
