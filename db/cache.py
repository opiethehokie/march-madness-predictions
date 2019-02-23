#   Copyright 2016-2019 Michael Peters
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import os
import sqlite3

import pandas as pd


def read_processed_data(name):
    return pd.read_sql('select * from %s;' % name, sqlite3.connect(_db_name(name)), index_col='index')

def write_processed_data(data, name):
    data.to_sql(name, sqlite3.connect(_db_name(name)), if_exists='replace')

def processed_data_exists(name):
    return os.path.isfile(_db_name(name))

def _db_name(name):
    return 'data/%s.db' % name
