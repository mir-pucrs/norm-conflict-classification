#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script creates embeddings from a file containing pairs of norms.
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from os.path import dirname

import embeddings
import filehandler as fh
import fileconfig

def main(normfile, output, wikiuni, jar, fasttext):
    fcfg = fileconfig.Configuration()
    fh.is_file(normfile)

    if output:
        fh.is_folder(output)
    else:
        output = dirname(normfile)

    if wikiuni:
        fh.is_file(wikiuni)
    else:
        wikiuni = fcfg.get('MODEL_WIKI_UNIGRAMS')

    if jar:
        fh.is_file(jar)
    else:
        jar = fcfg.get('SNLP_TAGGER_JAR')

    if fasttext:
        fh.is_file(fasttext)
    else:
        fasttext = fcfg.get('FASTTEXT_EXEC_PATH')

    embeddings.embeddings_from_norms(normfile, output, wikiuni, jar, fasttext)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('normfile', metavar='csv_with_norms', help='File containing pairs of norms')
    argparser.add_argument('-o', '--output', help='Folder to save embeddings', default=None)
    argparser.add_argument('-w', '--wikiunigram', help='File containing embeddings of unigrams from Wikipedia', default=None)
    argparser.add_argument('-j', '--jar', help='JAR file containing the Stanford tagger', default=None)
    argparser.add_argument('-f', '--fasttext', help='Path to the executable of FastText', default=None)
    args = argparser.parse_args()
    main(args.normfile, args.output, args.wikiunigram, args.jar, args.fasttext)
    

