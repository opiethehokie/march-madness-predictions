#   Copyright 2016-2020 Michael Peters
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

from feature_engine.outlier_removers import OutlierTrimmer

from db.cache import read_features, write_features, features_exist
from ml.aggregators import modified_rpi, statistics, custom_ratings, vanilla_stats, time_series_stats, descriptive_stats, elo
from ratings.off_def import adjust_stats
from ratings.markov import markov_stats


TOURNEY_START_DAY = 134

# cleaning

def filter_outlier_games(data, m=5):
    return OutlierTrimmer(distribution='gaussian', fold=m, tail='both').fit_transform(data)

def filter_overtime_games(data):
    return data[data['Numot'] == 0]

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

def oversample_neutral_site_games(data, factor=2):
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
    return np.clip(df.Lscore - df.Wscore, -25, 25)

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

# ETL

def prepare_data(games, future_games, start_day, start_year, predict_year):
    #print('Analyzing %d games' % len(games))
    games = filter_overtime_games(games)
    games = filter_outlier_games(games)
    games = oversample_neutral_site_games(games)
    games = concat_games(games, future_games)
    games = fill_missing_stats(games)
    features = extract_features(games, start_day)
    games, features = filter_out_of_window_games(games, features, start_day, start_year, predict_year)
    #print('Training on %d games, %d features' % (games.shape[0], features.shape[1]))
    #print('Feature list:\n', ['%i:%s' % (i, features.columns[i]) for i in range(0, len(features.columns))])
    assert games.shape[0] == features.shape[0]
    X_train, X_test, X_predict, y_train, y_test, cv = custom_train_test_split(games, features, predict_year)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    return X_train, X_test, X_predict, y_train, y_test, cv
