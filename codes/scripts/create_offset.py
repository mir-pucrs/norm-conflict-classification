#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
Creates a file containing the offset of all norm pairs from a CSV file 
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse

import embeddings
import filehandler as fh

def main(normfile, fembed, output):
    normfile = fh.is_file(normfile)
    
    if fembed:
        fembed = fh.is_file(fembed)
    else:
        fembed = fh.check_output(None, normfile, prefix='emb')
        fembed = fh.is_file(fembed)

    if output:
        output = fh.is_file(output)
    else:
        output = fh.check_output(None, normfile, prefix='offset')

    embeddings.generate_offset_from_file(normfile, fembed, output)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('normfile', metavar='csv_with_norms', help='File containing pairs of IDs of norms')
    argparser.add_argument('-e', '--embeddings', help='File containing embeddings of norms', default=None)
    argparser.add_argument('-o', '--output', help='File to save embeddings', default=None)
    args = argparser.parse_args()
    main(args.normfile, args.embeddings, args.output)
    

