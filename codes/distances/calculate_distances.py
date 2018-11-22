#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Calculate the distance of a unknown pair of norms to a cluster containing 
the mean offset embedding from conflicting norm pairs and to a cluster containing
the mean offset embedding from non-conflicting norm pairs.
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from os.path import join, dirname

from scripts import filehandler as fh
from scripts import embeddings
from scripts import fileconfig as fc
from scripts import utils

conf = fc.Configuration()


def check_distances(vpairs, vground, vdist, fout, distance):
    """ Save in a file the results """
    y_true, y_pred = [], []
    for norms, gtruth, arr in zip(vpairs, vground, vdist):
        if gtruth > 0: gtruth = 1
        norm1, norm2 = norms
        cft_cos, cft_euc, ncft_cos, ncft_euc = arr
        if distance == 'cos':
            if cft_cos < ncft_cos:
                fout.write('%d %d %d %d\n' % (norm1, norm2, conf.get('CONFLICT'), gtruth))
                y_true.append(gtruth)
                y_pred.append(conf.get('CONFLICT'))
            else:
                fout.write('%d %d %d %d\n' % (norm1, norm2, conf.get('N_CONFLICT'), gtruth))
                y_true.append(gtruth)
                y_pred.append(conf.get('N_CONFLICT'))
        elif distance == 'euc':
            if cft_euc < ncft_euc:
                fout.write('%d %d %d %d\n' % (norm1, norm2, conf.get('CONFLICT'), gtruth))
                y_true.append(gtruth)
                y_pred.append(conf.get('CONFLICT'))
            else:
                fout.write('%d %d %d %d\n' % (norm1, norm2, conf.get('N_CONFLICT'), gtruth))
                y_true.append(gtruth)
                y_pred.append(conf.get('N_CONFLICT'))
    return y_true, y_pred


def main(file_folds, cft_file, ncft_file, distance, output):
    # check input files
    file_folds = fh.is_file(file_folds)
    cft_file = fh.is_file(cft_file)
    ncft_file = fh.is_file(ncft_file)
    cft_emb = fh.check_output(None, cft_file, prefix='emb')
    ncft_emb = fh.check_output(None, ncft_file, prefix='emb')
    output = fh.is_file(output, boolean=True)
    if not output:
        output = join(dirname(cft_file), distance+'.txt')
    
    results = utils.KfoldResults()

    cft_norms = fh.NormPairsFile(cft_file)
    ncft_norms = fh.NormPairsFile(ncft_file)
    cft_dfemb = fh.DFrameEmbeddings(cft_emb)
    ncft_dfemb = fh.DFrameEmbeddings(ncft_emb)

    # process folds
    fout = open(output, 'w')
    fout.write('#id_1 id_2 pred ground\n')
    fds = fh.Folds(file_folds)
    for id_fold, arr in enumerate(fds):
        # train: tr :: test: ts
        # vp: vectors of pairs
        cft_tr, nn_tr, cft_ts, nn_ts = arr
        vp_cft_tr, vl_cft_tr = ids_from_indexes(cft_norms, cft_tr)
        vp_ncft_tr, vl_ncft_tr = ids_from_indexes(ncft_norms, nn_tr)
        vp_cft_ts, vl_cft_ts = ids_from_indexes(cft_norms, cft_ts)
        vp_ncft_ts, vl_ncft_ts = ids_from_indexes(ncft_norms, nn_ts)

        # compute offset for conflicts and non-conflicts
        cft_offset = embeddings.generate_offset(vp_cft_tr, cft_dfemb)
        ncft_offset = embeddings.generate_offset(vp_ncft_tr, ncft_dfemb)
        
        # generate distances for test set
        cft_dist = embeddings.calculate_distance_offset(vp_cft_ts, cft_dfemb, cft_offset, offset2=ncft_offset)
        ncft_dist = embeddings.calculate_distance_offset(vp_ncft_ts, ncft_dfemb, cft_offset, offset2=ncft_offset)

        fout.write('#fold: %d\n' % id_fold)
        # check distance for conflicts
        # pairs, ground, dist, fout, distance
        y_true, y_pred = check_distances(vp_cft_ts, vl_cft_ts, cft_dist, fout, distance)
        arr = check_distances(vp_ncft_ts, vl_ncft_ts, ncft_dist, fout, distance)
        y_true.extend(arr[0])
        y_pred.extend(arr[1])
        # check if the fold is the best one
        results.add(id_fold, y_true, y_pred)

    # add the mean of each measure to the results
    results.add_mean()

    # get best model and generate the offset
    best_fold, best_score = results.get_best(metric='accuracy')
    logger.info('Loading best fold: %d (accuracy: %f)' % (best_fold, best_score))
    cft_tr, nn_tr, _, _ = fds.get_fold(best_fold)
    cft_pairs_tr, _ = ids_from_indexes(cft_norms, cft_tr)
    ncft_pairs_tr, _ = ids_from_indexes(ncft_norms, nn_tr)

    cft_offset = embeddings.generate_offset(cft_pairs_tr, cft_dfemb)
    ncft_offset = embeddings.generate_offset(ncft_pairs_tr, ncft_dfemb)

    #load test set
    cft_ts, nn_ts = fds.test()
    cft_pairs_ts, cft_lbl_ts = ids_from_indexes(cft_norms, cft_ts)
    ncft_pairs_ts, ncft_lbl_ts = ids_from_indexes(ncft_norms, nn_ts)
    
    cft_dist = embeddings.calculate_distance_offset(cft_pairs_ts, cft_dfemb, cft_offset, offset2=ncft_offset)
    ncft_dist = embeddings.calculate_distance_offset(ncft_pairs_ts, ncft_dfemb, cft_offset, offset2=ncft_offset)

    fout.write('#fold: test\n')
    # check distance for conflicts
    # pairs, ground, dist, fout, distance
    y_true, y_pred = check_distances(cft_pairs_ts, cft_lbl_ts, cft_dist, fout, distance)
    arr = check_distances(ncft_pairs_ts, ncft_lbl_ts, ncft_dist, fout, distance)
    y_true.extend(arr[0])
    y_pred.extend(arr[1])
    # check if the fold is the best one
    results.add('test', y_true, y_pred)
    results.save(join(dirname(cft_file), 'results_'+distance+'.txt'))
    fout.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file_folds', metavar='json_folds', help='File containing train/test folds')
    argparser.add_argument('conflict_file', metavar='conflict_pairs', help='File containing pairs of norms with conflicts')
    argparser.add_argument('nonconflict_file', metavar='non_conflict_pairs', help='File containing pairs of norms without conflicts')
    argparser.add_argument('-o', '--output', help='File to save distances', default=None)
    argparser.add_argument('-d', '--distance', help='Distance to be measured', default='cos')
    args = argparser.parse_args()
    main(args.file_folds, args.conflict_file, args.nonconflict_file, args.distance, args.output)
    

