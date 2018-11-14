#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script will take all the images from inputfile.txt and classify according the neural net.
It will create an output file that have the path from the image, the correct label, and the predict label.
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import sys
import re
import time
from subprocess import call
import numpy as np
from nltk import TweetTokenizer
from nltk.tokenize.stanford import StanfordTokenizer
from os.path import isfile, abspath, join


def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence


def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token


def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]


def sent2embeddings(sentences, model_path, fasttext_exec_path):
    """Arguments:
        - sentences: a list of preprocessed sentences
        - model_path: a path to the sent2vec .bin model
        - fasttext_exec_path: a path to the fasttext executable
    """
    timestamp = str(time.time())
    test_path = abspath('./'+timestamp+'_fasttext.test.txt')
    embeddings_path = abspath('./'+timestamp+'_fasttext.embeddings.txt')
    dump_text_to_disk(test_path, sentences)
    call(fasttext_exec_path+
          ' print-sentence-vectors '+
          model_path + ' < '+
          test_path + ' > ' +
          embeddings_path, shell=True)
    embeddings = read_embeddings(embeddings_path)
    os.remove(test_path)
    os.remove(embeddings_path)
    print len(sentences), len(embeddings)
    assert(len(sentences) == len(embeddings))
    return np.array(embeddings)


def read_embeddings(embeddings_path):
    """Arguments:
        - embeddings_path: path to the embeddings
    """
    with open(embeddings_path, 'r') as in_stream:
        embeddings = []
        for line in in_stream:
            line = '['+line.replace(' ',',')+']'
            embeddings.append(eval(line))
        return embeddings
    return []


def dump_text_to_disk(file_path, X, Y=None):
    """Arguments:
        - file_path: where to dump the data
        - X: list of sentences to dump
        - Y: labels, if any
    """
    with open(file_path, 'w') as out_stream:
        if Y is not None:
            for x, y in zip(X, Y):
                out_stream.write('__label__'+str(y)+' '+x+' \n')
        else:
            for x in X:
                out_stream.write(x.encode('utf-8')+' \n')


def sentence_embeddings(wikiuni, snlpjar, fasttext, sentences, ngram='unigrams', model='concat_wiki_twitter'):
    """ 
    Generate embeddings from a list of sentences.

    Parameters:
    -----------
    wikiuni: string
        Path to the Wikipedia embeddings
    parser: string
        Path to the folder containing the Stanford Parser
    jar: string
        Path to the JAR file of the Stanford tagger
    fasttext: string
        Path to the executable of FastText 
    sentences: list
        List containing raw sentences
        e.g., ['Once upon a time', 'This is another sentence.', ...]
    ngram: string (unigram|bigram)
        ngram used in Wikipedia embeddings
    model: string (wiki|twitter|concat_wiki_twitter)
    """
    wiki_embeddings = None
    twitter_embbedings = None
    tokenized_sentences_NLTK_tweets = None
    tokenized_sentences_SNLP = None
    if model == "wiki" or model == 'concat_wiki_twitter':
        tknzr = StanfordTokenizer(snlpjar, encoding='utf-8')
        s = ' <delimiter> '.join(sentences) #just a trick to make things faster
        tkn_sentences_SNLP = tokenize_sentences(tknzr, [s])
        tkn_sentences_SNLP = tkn_sentences_SNLP[0].split(' <delimiter> ')
        assert(len(tkn_sentences_SNLP) == len(sentences))
        if ngram == 'unigrams':
            wiki_embeddings = sent2embeddings(tkn_sentences_SNLP, \
                                     wikiuni, fasttext)
    # We are not using Twitter or Bigrams so far
    #     else:
    #         wiki_embeddings = sent2embeddings(tkn_sentences_SNLP, \
    #                                  MODEL_WIKI_BIGRAMS, FASTTEXT_EXEC_PATH)
    # if model == "twitter" or model == 'concat_wiki_twitter':
    #     tknzr = TweetTokenizer()
    #     tkn_sentences_NLTK_tweets = tokenize_sentences(tknzr, sentences)
    #     if ngram == 'unigrams':
    #         twitter_embbedings = sent2embeddings(tkn_sentences_NLTK_tweets, \
    #                                  MODEL_TWITTER_UNIGRAMS, FASTTEXT_EXEC_PATH)
    #     else:
    #         twitter_embbedings = sent2embeddings(tkn_sentences_NLTK_tweets, \
    #                                  MODEL_TWITTER_BIGRAMS, FASTTEXT_EXEC_PATH)
    #
    if model == "wiki":
        return wiki_embeddings
    #elif model == "twitter":
    #    return twitter_embbedings
    #elif model == "concat_wiki_twitter":
    #    return np.concatenate((wiki_embeddings, twitter_embbedings), axis=1)
    sys.exit(-1)
