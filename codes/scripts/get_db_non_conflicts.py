#!/usr/bin/python
import sys
import random
import MySQLdb
import argparse
import progressbar
import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from deep_norm_conflict_identification.norm_identifier.sentence_classifier import SentenceClassifier

OUTPUT_FILE = 'db_non_conflicts.csv'
NORM_PROB = 0.8
CLASSIFIER_PATH = 'deep_norm_conflict_identification/norm_identifier/classifiers/16-10-25_12:18:39/sentence_classifier_16-10-25_12:18:39.pkl'
NAMES_PATH = 'deep_norm_conflict_identification/norm_identifier/classifiers/16-10-25_12:18:39/sentence_classifier_16-10-25_12:18:39_names.txt'
sent_cls = SentenceClassifier()
sent_cls.load_classifier(CLASSIFIER_PATH)
names = [n[:-1] for n in open(NAMES_PATH, 'r').readlines()]
vec = CountVectorizer(vocabulary=names)


def create_clause_pairs(df, con_id, norms, non_norms):

    if not norms or not non_norms:
        return 0

    # Run through pair of clauses and save them into the df.
    for i in range(len(norms)):
        # Randomly choose norms and non_norms to form pairs.
        prob_1 = random.random()
        
        if prob_1 > NORM_PROB:
            clause_id_1 = random.choice(non_norms.keys())
            clause_1 = non_norms[clause_id_1]
        else:
            clause_id_1 = random.choice(norms.keys())
            clause_1 = norms[clause_id_1]

        prob_2 = random.random()
        
        if prob_2 > NORM_PROB:
            clause_id_2 = random.choice(non_norms.keys())
            clause_2 = non_norms[clause_id_2]
        else:
            clause_id_2 = random.choice(norms.keys())
            clause_2 = norms[clause_id_2]

        # Fill in the dataframe.
        df['contract_id'].append(con_id)
        df['norm_id_1'].append(clause_id_1)
        df['norm_id_2'].append(clause_id_2)
        df['norm1'].append(clause_1)
        df['norm2'].append(clause_2)


def main(user, passwd, database):

    # Open database connection
    db = MySQLdb.connect("localhost", user, passwd, database)

    # Prepare a cursor object using cursor() method.
    cursor = db.cursor()

    print "Selecting clauses ...."
    # Retrieve all created conflicts.
    sel_clauses = """SELECT cl.con_id, cl.clause_id, cl.clause_range
               FROM clauses as cl, conflicts as con
               WHERE cl.clause_id != con.clause_id_1
               AND cl.clause_id != con.clause_id_2
               GROUP BY cl.clause_id"""

    cursor.execute(sel_clauses)
    clause_tups = cursor.fetchall()

    print 'Selected %d clauses' % len(clause_tups)

    clause_ranges = dict() # Create the dictionary to save clauses.

    df = OrderedDict() # Create dataframe.
    df['contract_id'] = list()
    df['norm_id_1'] = list()
    df['norm_id_2'] = list()
    df['norm1'] = list()
    df['norm2'] = list()

    print "Selecting ranges."
    # Separate clauses by contracts.
    for clause_tup in clause_tups:
        # Retrieve elements.
        con_id = clause_tup[0]
        clause_id = clause_tup[1]
        clause_range = clause_tup[2]

        if con_id not in clause_ranges:
            clause_ranges[con_id] = [(clause_id, clause_range)]
        else:
            clause_ranges[con_id].append((clause_id, clause_range))

    pb = progressbar.ProgressBar(maxval=len(clause_ranges.keys())).start()

    print "Stating the contract process."
    
    for con_id in clause_ranges:
        sel_contract = """SELECT path_to_file
                          FROM contracts
                          WHERE con_id=%s""" % con_id
        cursor.execute(sel_contract)
        contract_path = cursor.fetchone()[0]
        # Read contract.
        con_text = open(contract_path, 'r').read()

        norms = dict()
        non_norms = dict()

        for range_tup in clause_ranges[con_id]:

            # Get the range of text.
            ranges = range_tup[1].strip('()').split(',')

            clause = con_text[int(ranges[0]):int(ranges[1])]
            # Check if it is a norm using the classifier.
            if clause:
                t = vec.fit_transform([clause])
                clss = sent_cls.classifier.predict(t.A)

                if clss[0] == 1:
                    norms[range_tup[0]] = clause
                else:
                    non_norms[range_tup[0]] = clause
        create_clause_pairs(df, con_id, norms, non_norms)
        
        pb.update()

    # Save dataframe.
    df = pd.DataFrame(data=df)
    df.to_csv(OUTPUT_FILE, index=False)

    # Close DB.
    db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db_user", help="DB username.")
    parser.add_argument("passwd", help="DB password for the user.")
    parser.add_argument("database", help="DB name to connect.")
    parser.add_argument("--output_file", help="Path name to the output file.")

    args = parser.parse_args()
    if args.output_file:
        OUTPUT_FILE = args.output_file

    main(args.db_user, args.passwd, args.database)    