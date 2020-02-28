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


import numpy as np
import pandas as pd
import scipy

from db.cache import read_features, write_features, features_exist
from ml2.aggregators import modified_rpi, statistics, custom_ratings, vanilla_stats, time_series_stats, descriptive_stats, elo
from ratings.off_def import adjust_stats
from ratings.markov import markov_stats


TOURNEY_START_DAY = 134

# cleaning

def filter_outlier_games(data, m=5):
    numeric_data = data.select_dtypes(include=['int64'])
    return data[(np.abs(scipy.stats.zscore(numeric_data)) < m).all(axis=1)]

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

def concat_games(data1, data2):
    return pd.concat([data1, data2], axis=0, sort=False, ignore_index=True)

def fill_missing_stats(data):
    return data.fillna(0)

def filter_out_of_window_games(data, features, sday, syear, eyear):
    in_window_data = (data.pipe(lambda df: df[df.Daynum >= sday])
                      .pipe(lambda df: df[df.Season >= syear])
                      .pipe(lambda df: df[df.Season <= eyear]))
    in_window_features = features.query('Season >= @syear and Season <= @eyear')
    return in_window_data, in_window_features

# sampling

def oversample_neutral_site_games(data, factor=3):
    data = data.copy()
    neutral_site_games = data[(data.Wloc == 'N') & (data.Daynum < TOURNEY_START_DAY)]
    return data.append([neutral_site_games]*factor, ignore_index=True)

def custom_train_test_split(data, features, predict_year):
    train_games = data[(data.Season != predict_year) | (data.Daynum < TOURNEY_START_DAY)]
    train_features = features.query('Season != @predict_year or Daynum < @TOURNEY_START_DAY')
    test_games = data[(data.Season == predict_year) & (data.Daynum >= TOURNEY_START_DAY) & (data.Daynum != 999)]
    test_features = features.query('Season == @predict_year and Daynum >= @TOURNEY_START_DAY and Daynum != 999')
    predict_features = features.query('Season == @predict_year and Daynum == 999')
    train_results = train_games[['Wteam', 'Lteam']].apply(_win, axis=1)
    test_results = test_games[['Wteam', 'Lteam']].apply(_win, axis=1)
    cv = _custom_cv(train_features)
    return (train_features.values.astype('float64'), test_features.values.astype('float64'), predict_features.values.astype('float64'),
            train_results.values, test_results.values, cv)

# sort of walk-forward cross-validation
# https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9
def _custom_cv(X):
    season_idx = X.index.get_level_values('Season')
    seasons = np.sort(season_idx.unique())
    day_idx = X.index.get_level_values('Daynum')
    return [(np.where((season_idx == season) & (day_idx < TOURNEY_START_DAY))[0],
             np.where((season_idx == season) & (day_idx >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]

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

def _construct_sos(data, start_day, bust_cache=False):
    if features_exist('sos') and not bust_cache:
        return read_features('sos')
    rpi1 = pd.DataFrame(modified_rpi(data, start_day, weights=(.15, .15, .7)), columns=['rpi1'])
    rpi2 = pd.DataFrame(modified_rpi(data, start_day, weights=(.25, .25, .5)), columns=['rpi2'])
    rpi3 = pd.DataFrame(modified_rpi(data, start_day, weights=(.25, .5, .25)), columns=['rpi3'])
    sos = pd.concat([rpi1, rpi2, rpi3], axis=1)
    write_features(sos, 'sos')
    return sos

def _construct_stats(data, start_day, bust_cache=False):
    if features_exist('stats') and not bust_cache:
        return read_features('stats')
    stats1 = pd.DataFrame(statistics(data, start_day, vanilla_stats))
    stats1.columns = ['stat%s' % i for i in range(1, np.size(stats1, 1) + 1)]
    stats2 = pd.DataFrame(statistics(data, start_day, descriptive_stats))
    stats2.columns = ['descstat%s' % i for i in range(1, np.size(stats2, 1) + 1)]
    stats3 = pd.DataFrame(statistics(data, start_day, time_series_stats))
    stats3.columns = ['timeseriesstat%s' % i for i in range(1, np.size(stats3, 1) + 1)]
    stats = pd.concat([stats1, stats2, stats3], axis=1)
    write_features(stats, 'stats')
    return stats

def _construct_ratings(data, start_day, bust_cache=False):
    if features_exist('ratings') and not bust_cache:
        return read_features('ratings')
    rating1 = pd.DataFrame(custom_ratings(data, start_day, adjust_stats), columns=['off', 'def'])
    rating2 = pd.DataFrame(custom_ratings(data, start_day, markov_stats), columns=['markov'])
    rating3 = pd.DataFrame(elo(data, start_day), columns=['elo'])
    ratings = pd.concat([rating1, rating2, rating3], axis=1)
    write_features(ratings, 'ratings')
    return ratings

def extract_features(data, start_day):
    sos = _construct_sos(data, start_day)
    stats = _construct_stats(data, start_day)
    ratings = _construct_ratings(data, start_day)
    features = pd.concat([sos.reset_index(drop=True),
                          stats.reset_index(drop=True),
                          ratings.reset_index(drop=True)
                         ], axis=1)
    feature_data = data[(data.Daynum >= start_day)]
    features.index = pd.MultiIndex.from_arrays(feature_data[['Season', 'Daynum']].values.T, names=['Season', 'Daynum'])
    return features
