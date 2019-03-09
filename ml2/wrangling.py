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


import itertools
import numpy as np
import pandas as pd
import scipy as sp

from db.cache import read_features, write_features, features_exist
from ml2.aggregators import modified_rpi, statistics, custom_ratings
from ratings.off_def import adjust_stats
from ratings.markov import markov_stats


TOURNEY_START_DAY = 134

# cleaning

def filter_outlier_games(data, m=5):
    numeric_data = data.select_dtypes(include=['int64'])
    return data[(np.abs(sp.stats.zscore(numeric_data)) < m).all(axis=1)]

def adjust_overtime_games(data):
    data = data.copy()
    for row in data.itertuples(index=True):
        if row.Numot > 0:
            ot_adj = 40 / (40 + row.Numot * 5)
            data.at[row.Index, 'Wscore'] = row.Wscore * ot_adj
            data.at[row.Index, 'Lscore'] = row.Lscore * ot_adj
            data.at[row.Index, 'Wfgm'] = row.Wfgm * ot_adj
            data.at[row.Index, 'Lfgm'] = row.Lfgm * ot_adj
            data.at[row.Index, 'Wfga'] = row.Wfga * ot_adj
            data.at[row.Index, 'Lfga'] = row.Lfga * ot_adj
            data.at[row.Index, 'Wfgm3'] = row.Wfgm3 * ot_adj
            data.at[row.Index, 'Lfgm3'] = row.Lfgm3 * ot_adj
            data.at[row.Index, 'Wfga3'] = row.Wfga3 * ot_adj
            data.at[row.Index, 'Lfga3'] = row.Lfga3 * ot_adj
            data.at[row.Index, 'Wftm'] = row.Wftm * ot_adj
            data.at[row.Index, 'Lftm'] = row.Lftm * ot_adj
            data.at[row.Index, 'Wfta'] = row.Wfta * ot_adj
            data.at[row.Index, 'Lfta'] = row.Lfta * ot_adj
            data.at[row.Index, 'Wor'] = row.Wor * ot_adj
            data.at[row.Index, 'Lor'] = row.Lor * ot_adj
            data.at[row.Index, 'Wdr'] = row.Wdr * ot_adj
            data.at[row.Index, 'Ldr'] = row.Ldr * ot_adj
            data.at[row.Index, 'Wast'] = row.Wast * ot_adj
            data.at[row.Index, 'Last'] = row.Last * ot_adj
            data.at[row.Index, 'Wto'] = row.Wto * ot_adj
            data.at[row.Index, 'Lto'] = row.Lto * ot_adj
            data.at[row.Index, 'Wstl'] = row.Wstl * ot_adj
            data.at[row.Index, 'Lstl'] = row.Lstl * ot_adj
            data.at[row.Index, 'Wblk'] = row.Wblk * ot_adj
            data.at[row.Index, 'Lblk'] = row.Lblk * ot_adj
            data.at[row.Index, 'Wpf'] = row.Wpf * ot_adj
            data.at[row.Index, 'Lpf'] = row.Lpf * ot_adj
    return data

def add_games(data1, data2):
    return pd.concat([data1, data2], axis=0, sort=False, ignore_index=True)

def fill_missing_stats(data):
    return data.fillna(0)

def filter_out_of_window_games(data, syear, sday, eyear):
    season = (data.pipe(lambda df: df[df.Season >= syear])
              .pipe(lambda df: df[df.Season <= eyear])
              .pipe(lambda df: df[df.Daynum >= sday]))
    return season

# sampling

def oversample_neutral_site_games(data, factor=3):
    data = data.copy()
    neutral_site_games = data[(data.Wloc == 'N') & (data.Daynum < TOURNEY_START_DAY)]
    return data.append([neutral_site_games]*factor, ignore_index=True)

def custom_train_test_split(data, predict_year, drop=True):
    train_games = data[(data.Season != predict_year) | (data.Daynum < TOURNEY_START_DAY)]
    test_games = data[(data.Season == predict_year) & (data.Daynum >= TOURNEY_START_DAY) & (data.Daynum != 999)]
    predict_games = data[(data.Season == predict_year) & (data.Daynum == 999)]
    train_results = train_games[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(_mov, axis=1)
    test_results = test_games[['Wteam', 'Lteam']].apply(_win, axis=1)
    not_needed_cols = ['Daynum', 'Season', 'Wteam', 'Lteam', 'Wscore', 'Lscore', 'Wloc', 'Numot',
                       'Wfgm', 'Wfga', 'Wfgm3', 'Wfga3', 'Wftm', 'Wfta', 'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf',
                       'Lfgm', 'Lfga', 'Lfgm3', 'Lfga3', 'Lftm', 'Lfta', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf']
    if drop:
        train_games = train_games.drop(columns=not_needed_cols)
        test_games = test_games.drop(columns=not_needed_cols)
        predict_games = predict_games.drop(columns=not_needed_cols)
    return (train_games.values.astype('float64'), test_games.values.astype('float64'), predict_games.values.astype('float64'),
            train_results.values, test_results.values)

def custom_cv(X):
    season_col = X[:, 0]
    seasons = np.unique(season_col)
    day_col = X[:, 1]
    return [(np.where((season_col != season) & (day_col < TOURNEY_START_DAY))[0],
             np.where((season_col == season) & (day_col >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]

def _win(df):
    return int(df.Wteam < df.Lteam)

def _mov(df):
    if df.Wteam < df.Lteam:
        return df.Wscore - df.Lscore
    return df.Lscore - df.Wscore

# data stats

def tourney_mov_std(data):
    tourney_games = data[(data.Daynum >= TOURNEY_START_DAY)]
    movs = tourney_games[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(_mov, axis=1)
    return np.std(movs)

# feature extraction

def _construct_sos(data, bust_cache=False):
    if features_exist('sos') and not bust_cache:
        return read_features('sos')
    rpi1 = pd.DataFrame(modified_rpi(data, weights=(.15, .15, .7)), columns=['rpi_1a', 'rpi_1b'])
    rpi2 = pd.DataFrame(modified_rpi(data, weights=(.25, .25, .5)), columns=['rpi_2a', 'rpi_2b'])
    rpi3 = pd.DataFrame(modified_rpi(data, weights=(.25, .5, .25)), columns=['rpi_3a', 'rpi_3b'])
    sos = pd.concat([rpi1, rpi2, rpi3], axis=1)
    write_features(sos, 'sos')
    return sos

def _construct_stats(data, bust_cache=False):
    if features_exist('stats') and not bust_cache:
        return read_features('stats')
    raw = statistics(data)
    num_stats = int(np.size(raw, 1) / 2)
    a_cols = ['stat_%sa' % i for i in range(1, num_stats + 1)]
    b_cols = ['stat_%sb' % i for i in range(1, num_stats + 1)]
    stats = pd.DataFrame(raw, columns=[x for x in itertools.chain.from_iterable(itertools.zip_longest(a_cols, b_cols)) if x])
    write_features(stats, 'stats')
    return stats

def _construct_ratings(data, bust_cache=False):
    if features_exist('ratings') and not bust_cache:
        return read_features('ratings')
    offdef = pd.DataFrame(custom_ratings(data, adjust_stats), columns=['off_a', 'off_b', 'def_a', 'def_b'])
    markov = pd.DataFrame(custom_ratings(data, markov_stats), columns=['markov_a', 'markov_b'])
    ratings = pd.concat([offdef, markov], axis=1)
    write_features(ratings, 'ratings')
    return ratings

def extract_features(data):
    sos = _construct_sos(data)
    stats = _construct_stats(data)
    ratings = _construct_ratings(data)
    return pd.concat([data.reset_index(drop=True),
                      sos.reset_index(drop=True),
                      stats.reset_index(drop=True),
                      ratings.reset_index(drop=True)
                     ], axis=1)
