import csv
import os
import sqlite3

import pandas as pd

from joblib import dump, load


def read_features(name):
    return pd.read_sql('select * from %s;' % name, sqlite3.connect(_db_name(name)), index_col='index')

def write_features(features, name):
    features.to_sql(name, sqlite3.connect(_db_name(name)), if_exists='replace')

def features_exist(name):
    return os.path.isfile(_db_name(name))

def _db_name(name):
    return 'data/%s.db' % name

def save_meta_data(model, score, train_data_shape, hyperparameters, random_seed):
    meta_data_name = 'data/metadata.csv'
    with open(meta_data_name, 'a', newline='') as mdf:
        writer = csv.writer(mdf, lineterminator='\n')
        writer.writerow([model, train_data_shape, score, hyperparameters, random_seed])
