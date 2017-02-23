#   Copyright 2016 Michael Peters
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


import cProfile
import itertools
import numpy
import scipy.stats


TOURNEY_START_DAY = 136

def all_teams(data):
    seasons = data['Season'].unique()
    teams = {}
    for season in seasons:
        season_games = data.query('Season == %s' % season)
        sorted_teams = sorted((season_games['Wteam'].append(season_games['Lteam'])).unique())
        teams[season] = {x:i for i, x in enumerate(sorted_teams)}
    return teams

def oversample_tourney_games(data, factor=5):
    tourney_games = data[data.Daynum > TOURNEY_START_DAY]
    return data.append([tourney_games]*factor, ignore_index=True)

def filter_outlier_games(data, m=3):
    numeric_data = data.select_dtypes(include=['int64'])
    return data[(numpy.abs(scipy.stats.zscore(numeric_data)) < m).all(axis=1)]

def custom_train_test_split(data, predict_year):
    train_games = data[(data.Season != predict_year) | (data.Daynum < TOURNEY_START_DAY)]
    holdout_games = data[(data.Season == predict_year) & (data.Daynum >= TOURNEY_START_DAY)]
    def encode_winner(df):
        return int(df.Wteam < df.Lteam)
    train_results = train_games[['Wteam', 'Lteam']].apply(encode_winner, axis=1)
    holdout_results = holdout_games[['Wteam', 'Lteam']].apply(encode_winner, axis=1)
    return train_games, holdout_games, train_results, holdout_results

def derive_stats(results):
    derived = []
    combinations = itertools.combinations(results.keys(), 2)
    for stat1, stat2 in combinations:
        avg_stat1 = sum(results[stat1]) / len(results[stat1])
        avg_stat2 = sum(results[stat2]) / len(results[stat2])
        derived.append(avg_stat1 / avg_stat2 if avg_stat2 > 0 else 0)
    return derived

def describe_stats(results):
    described = []
    for stat in results.keys():
        described.append(min(results[stat]))
        described.append(max(results[stat]))
        described.append(numpy.std(results[stat]))
        described.append(numpy.median(results[stat]))
        described.append(numpy.mean(results[stat]))
    return described

# annotate a function with @profile to see where it's spending the most time
def profile(func):
    def profiled_func(*args, **kwargs):
        p = cProfile.Profile()
        try:
            p.enable()
            result = func(*args, **kwargs)
            p.disable()
            return result
        finally:
            p.print_stats()
    return profiled_func
