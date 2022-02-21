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

def read_model(name):
    return load(_dump_name(name))

def write_model(model, name):
    dump(model, _dump_name(name))

def model_exists(name):
    return os.path.isfile(_dump_name(name))

def _dump_name(name):
    return 'data/%s.joblib' % name
