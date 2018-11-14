#!/usr/bin/python
import sys
import MySQLdb
import argparse
import progressbar
import pandas as pd
from collections import OrderedDict

OUTPUT_FILE = 'db_conflicts.csv'

def main(user, passwd, database):

    # Open database connection
    db = MySQLdb.connect("localhost", user, passwd, database)

    # Prepare a cursor object using cursor() method.
    cursor = db.cursor()

    # Retrieve all created conflicts.
    query = """SELECT con_id, conf_id, clause_id_1, clause_id_2, type_id
               FROM conflicts
               WHERE classifier_id is NULL"""

    cursor.execute(query)
    clauses_tup = cursor.fetchall()

    # Open file to write.
    w_file = open(OUTPUT_FILE, 'w')
    # Write header.
    d = OrderedDict()
    d['conflict_id'] = list()
    d['contract_id'] = list()
    d['norm_id_1'] = list()
    d['norm_id_2'] = list()
    d['norm1'] = list()
    d['norm2'] = list()
    d['conf_type'] = list()
    
    # Fetch a single row using fetchone() method.
    for tup in clauses_tup:
        con_id = tup[0]
        conf_id = tup[1]
        clause_id_1 = tup[2]
        clause_id_2 = tup[3]
        type_id = tup[4]
       
        if not type_id:
            type_id = 1
        elif int(type_id) == 2:
            continue

        # Get contract path.
        cntrct_path_query = """SELECT path_to_file
                               FROM contracts
                               WHERE con_id=%d""" % con_id
        cursor.execute(cntrct_path_query)
        contract_path = cursor.fetchone()[0]

        # Get contract text.
        contract_text = open(contract_path, 'r').read()
        # Get the range for clause 1.
        rng_1_query = """SELECT clause_range
                         FROM clauses
                         WHERE clause_id=%d""" % clause_id_1
        cursor.execute(rng_1_query)
        clause_1_range = cursor.fetchone()[0]
        clause_1_range = clause_1_range.strip('()').split(',')
        # Get the range for clause 2.
        rng_2_query = """SELECT clause_range
                         FROM clauses
                         WHERE clause_id=%d""" % clause_id_2
        cursor.execute(rng_2_query)
        clause_2_range = cursor.fetchone()[0]
        clause_2_range = clause_2_range.strip('()').split(',')
        
        # Get clause texts.
        clause_1 = contract_text[int(clause_1_range[0]):int(clause_1_range[1])]
        clause_2 = contract_text[int(clause_2_range[0]):int(clause_2_range[1])]
        
        # Store clause pair to a list.
        if clause_1 and clause_2:
            d['contract_id'].append(con_id)
            d['conflict_id'].append(conf_id)
            d['norm_id_1'].append(clause_id_1)
            d['norm_id_2'].append(clause_id_2)
            d['norm1'].append(clause_1)
            d['norm2'].append(clause_2)
            d['conf_type'].append(type_id)

    # Disconnect from database.
    db.close()

    df = pd.DataFrame(data=d)
    df.to_csv(OUTPUT_FILE, index=False)
    print "Conflicts gathered and saved at %s" % OUTPUT_FILE

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
