#   Copyright 2016-2018 Michael Peters
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


from collections import defaultdict
from functools import partial

import itertools
import numpy
import pandas
import scipy.stats

from ratings.off_def import adjust_stats
from ratings.markov import markov_stats


TOURNEY_START_DAY = 136

# pre-processing

def filter_outlier_games(data, m=3):
    numeric_data = data.select_dtypes(include=['int64'])
    return data[(numpy.abs(scipy.stats.zscore(numeric_data)) < m).all(axis=1)]

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

def create_synthetic_games(data, home_factor=.96, neutral_site_factor=1):
    data = data.copy()
    # adjust for home court advantage
    for row in data.itertuples(index=True):
        if row.Wloc != 'N':
            wadj = home_factor if row.Wloc == 'H' else 1 / home_factor
            ladj = 2 - home_factor if row.Wloc == 'H' else home_factor
            data.at[row.Index, 'Wscore'] = row.Wscore * wadj
            data.at[row.Index, 'Lscore'] = row.Lscore * ladj
            data.at[row.Index, 'Wfgm'] = row.Wfgm * wadj
            data.at[row.Index, 'Lfgm'] = row.Lfgm * ladj
            data.at[row.Index, 'Wfga'] = row.Wfga * wadj
            data.at[row.Index, 'Lfga'] = row.Lfga * ladj
            data.at[row.Index, 'Wfgm3'] = row.Wfgm3 * wadj
            data.at[row.Index, 'Lfgm3'] = row.Lfgm3 * ladj
            data.at[row.Index, 'Wfga3'] = row.Wfga3 * wadj
            data.at[row.Index, 'Lfga3'] = row.Lfga3 * ladj
            data.at[row.Index, 'Wftm'] = row.Wftm * wadj
            data.at[row.Index, 'Lftm'] = row.Lftm * ladj
            data.at[row.Index, 'Wfta'] = row.Wfta * wadj
            data.at[row.Index, 'Lfta'] = row.Lfta * ladj
            data.at[row.Index, 'Wor'] = row.Wor * wadj
            data.at[row.Index, 'Lor'] = row.Lor * ladj
            data.at[row.Index, 'Wdr'] = row.Wdr * wadj
            data.at[row.Index, 'Ldr'] = row.Ldr * ladj
            data.at[row.Index, 'Wast'] = row.Wast * wadj
            data.at[row.Index, 'Last'] = row.Last * ladj
            data.at[row.Index, 'Wto'] = row.Wto * wadj
            data.at[row.Index, 'Lto'] = row.Lto * ladj
            data.at[row.Index, 'Wstl'] = row.Wstl * wadj
            data.at[row.Index, 'Lstl'] = row.Lstl * ladj
            data.at[row.Index, 'Wblk'] = row.Wblk * wadj
            data.at[row.Index, 'Lblk'] = row.Lblk * ladj
            data.at[row.Index, 'Wpf'] = row.Wpf * wadj
            data.at[row.Index, 'Lpf'] = row.Lpf * ladj
    # oversample neutral site games
    neutral_site_games = data[data.Wloc == 'N']
    return data.append([neutral_site_games]*neutral_site_factor, ignore_index=True)

def custom_train_test_split(data, predict_year):
    train_games = data[(data.Season != predict_year) | (data.Daynum < TOURNEY_START_DAY)]
    holdout_games = data[(data.Season == predict_year) & (data.Daynum >= TOURNEY_START_DAY)]
    def mov(df):
        if df.Wteam < df.Lteam:
            return df.Wscore - df.Lscore # 1 for classification
        return df.Lscore - df.Wscore # 0 for classification
    def win(df):
        return int(df.Wteam < df.Lteam)
    train_results = train_games[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(mov, axis=1)
    holdout_results = holdout_games[['Wteam', 'Lteam', 'Wscore', 'Lscore']].apply(win, axis=1)
    not_needed_cols = ['Wscore', 'Lscore', 'Wloc', 'Numot', 'Wfgm', 'Wfga', 'Wfgm3', 'Wfga3', 'Wftm', 'Wfta',
                       'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf', 'Lfgm', 'Lfga', 'Lfgm3', 'Lfga3', 'Lftm',
                       'Lfta', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf']
    train_games = train_games.drop(columns=not_needed_cols)
    holdout_games = holdout_games.drop(columns=not_needed_cols)
    return train_games.values.astype('float64'), holdout_games.values.astype('float64'), train_results.values, holdout_results.values

# data prep

def derive_stats(results):
    derived = []
    combinations = itertools.combinations(results.keys(), 2)
    for stat1, stat2 in combinations:
        avg_stat1 = sum(results[stat1]) / len(results[stat1]) if results[stat1] else 0
        avg_stat2 = sum(results[stat2]) / len(results[stat2]) if results[stat2] else 0
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
        #described.append(scipy.stats.kurtosis(results[stat], fisher=False))
        #described.append(scipy.stats.skew(results[stat]))
    return described

def _rpi(season_results, team, weights):
    results = season_results[team]['results']
    win_percent = sum(results) / len(results) if results else 0
    opponents_win_percent = _opponents_win_percent(season_results, [o for o in season_results[team]['opponents'] if o != team])
    opponents_opponents_win_percent = _opponents_opponents_win_percent(season_results, season_results[team]['opponents'])
    w1, w2, w3 = weights[0], weights[1], weights[2]
    return w1 * win_percent + w2 * opponents_win_percent + w3 * opponents_opponents_win_percent

def _opponents_win_percent(season_results, opponents):
    win_percents = []
    for opponent in opponents:
        results = season_results[opponent]['results']
        win_percent = sum(results) / len(results) if results else 0
        win_percents.append(win_percent)
    return sum(win_percents) / len(opponents) if opponents else 0

def _opponents_opponents_win_percent(season_results, opponents):
    win_percents = []
    for opponent in opponents:
        win_percent = _opponents_win_percent(season_results, season_results[opponent]['opponents'])
        win_percents.append(win_percent)
    return sum(win_percents) / len(opponents) if opponents else 0

def modified_rpi(X, X_pred, weights=(.15, .15, .7)):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    rpis = []
    for row in X.itertuples(index=False):
        stats[row.Season][row.Wteam]['opponents'].append(row.Lteam)
        stats[row.Season][row.Lteam]['opponents'].append(row.Wteam)
        stats[row.Season][row.Wteam]['results'].append(1)
        stats[row.Season][row.Lteam]['results'].append(0)
        wrpi = _rpi(stats[row.Season], row.Wteam, weights)
        lrpi = _rpi(stats[row.Season], row.Lteam, weights)
        if row.Wteam < row.Lteam:
            rpis.append([wrpi, lrpi])
        else:
            rpis.append([lrpi, wrpi])
    rpis_pred = []
    for row in X_pred.itertuples(index=False):
        wrpi = _rpi(stats[row.Season], row.Wteam, weights)
        lrpi = _rpi(stats[row.Season], row.Lteam, weights)
        if row.Wteam < row.Lteam:
            rpis_pred.append([wrpi, lrpi])
        else:
            rpis_pred.append([lrpi, wrpi])
    return numpy.array(rpis), numpy.array(rpis_pred)

def _pythagorean_expectation(results, exponent):
    points = numpy.sum(results['points'])
    points_against = numpy.sum(results['points-against'])
    numerator = points**exponent
    denomenator = points**exponent + points_against**exponent
    return numerator / denomenator if denomenator != 0 else 0

def pythagorean_expectation(X, X_pred, exponent=10.25):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    pythags = []
    for row in X.itertuples(index=False):
        stats[row.Season][row.Wteam]['points'].append(row.Wscore)
        stats[row.Season][row.Wteam]['points-against'].append(row.Lscore)
        stats[row.Season][row.Lteam]['points'].append(row.Lscore)
        stats[row.Season][row.Lteam]['points-against'].append(row.Wscore)
        wpythag = _pythagorean_expectation(stats[row.Season][row.Wteam], exponent)
        lpythag = _pythagorean_expectation(stats[row.Season][row.Lteam], exponent)
        if row.Wteam < row.Lteam:
            pythags.append([wpythag, lpythag])
        else:
            pythags.append([lpythag, wpythag])
    pythags_pred = []
    for row in X_pred.itertuples(index=False):
        wpythag = _pythagorean_expectation(stats[row.Season][row.Wteam], exponent)
        lpythag = _pythagorean_expectation(stats[row.Season][row.Lteam], exponent)
        if row.Wteam < row.Lteam:
            pythags_pred.append([wpythag, lpythag])
        else:
            pythags_pred.append([lpythag, wpythag])
    return numpy.array(pythags), numpy.array(pythags_pred)

def _all_teams(data):
    seasons = data['Season'].unique()
    teams = {}
    for season in seasons:
        season_games = data.query('Season == %s' % season)
        sorted_teams = sorted((season_games['Wteam'].append(season_games['Lteam'])).unique())
        teams[season] = {X:i for i, X in enumerate(sorted_teams)}
    return teams

def advanced_statistic_ratings(X, X_pred, rating_F, preseason_games=None):
    first_day = min(numpy.unique(X['Daynum']))
    if preseason_games is not None:
        X = pandas.concat([preseason_games, X], ignore_index=True)
    teams = _all_teams(X)
    seasons = numpy.unique(X['Season'])
    stat_categories = {X:i for i, X in enumerate(['eff_field_goal_percent', 'true_shooting', 'rebound_rate', 'free_throw_rate',
                                                  'turnover_rate', 'assist_rate', 'block_rate', 'steal_rate', 'score_rate', 'foul_rate'])}
    stats = {season: numpy.zeros((len(teams[season]), len(teams[season]), len(stat_categories))) for season in seasons}
    ratings = []
    day = 0
    adj_stats = None
    for row in X.itertuples(index=False):
        wteam_id = teams[row.Season][row.Wteam]
        lteam_id = teams[row.Season][row.Lteam]
        wposs = row.Wfga - row.Wor + row.Wto + .475 * row.Wfta
        lposs = row.Lfga - row.Lor + row.Lto + .475 * row.Lfta
        stats[row.Season][wteam_id][lteam_id][stat_categories['eff_field_goal_percent']] += ((row.Wfgm + .5 * row.Wfgm3) / row.Wfga)
        stats[row.Season][lteam_id][wteam_id][stat_categories['eff_field_goal_percent']] += ((row.Lfgm + .5 * row.Lfgm3) / row.Lfga)
        stats[row.Season][wteam_id][lteam_id][stat_categories['true_shooting']] += (.5 * row.Wscore / (row.Wfga + 0.475 * row.Wfta))
        stats[row.Season][lteam_id][wteam_id][stat_categories['true_shooting']] += (.5 * row.Lscore / (row.Lfga + 0.475 * row.Lfta))
        stats[row.Season][wteam_id][lteam_id][stat_categories['rebound_rate']] += row.Wor / (row.Wor + row.Ldr)
        stats[row.Season][lteam_id][wteam_id][stat_categories['rebound_rate']] += row.Lor / (row.Lor + row.Wdr)
        stats[row.Season][wteam_id][lteam_id][stat_categories['free_throw_rate']] += (row.Wfta / row.Wftm) if row.Wftm > 0 else 0
        stats[row.Season][lteam_id][wteam_id][stat_categories['free_throw_rate']] += (row.Lfta / row.Lftm) if row.Lftm > 0 else 0
        stats[row.Season][wteam_id][lteam_id][stat_categories['turnover_rate']] += (row.Wto / wposs)
        stats[row.Season][lteam_id][wteam_id][stat_categories['turnover_rate']] += (row.Lto / lposs)
        stats[row.Season][wteam_id][lteam_id][stat_categories['assist_rate']] += (row.Wast / row.Wfgm)
        stats[row.Season][lteam_id][wteam_id][stat_categories['assist_rate']] += (row.Last / row.Lfgm)
        stats[row.Season][wteam_id][lteam_id][stat_categories['block_rate']] += (row.Wblk / row.Lfga)
        stats[row.Season][lteam_id][wteam_id][stat_categories['block_rate']] += (row.Lblk / row.Wfga)
        stats[row.Season][wteam_id][lteam_id][stat_categories['steal_rate']] += (row.Wstl / lposs)
        stats[row.Season][lteam_id][wteam_id][stat_categories['steal_rate']] += (row.Lstl / wposs)
        stats[row.Season][wteam_id][lteam_id][stat_categories['score_rate']] += (row.Wscore / wposs)
        stats[row.Season][lteam_id][wteam_id][stat_categories['score_rate']] += (row.Lscore / lposs)
        stats[row.Season][wteam_id][lteam_id][stat_categories['foul_rate']] += (row.Wpf / lposs)
        stats[row.Season][lteam_id][wteam_id][stat_categories['foul_rate']] += (row.Lpf / wposs)
        if row.Daynum >= first_day:
            if row.Daynum > day:
                adj_stats = rating_F(stats)
                day = row.Daynum
            team_ids = teams[row.Season]
            wrating = adj_stats[row.Season][team_ids[row.Wteam]]
            lrating = adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                ratings.append(numpy.hstack((wrating, lrating)))
            else:
                ratings.append(numpy.hstack((lrating, wrating)))
    ratings_pred = []
    for row in X_pred.itertuples(index=False):
        team_ids = teams[row.Season]
        wrating = adj_stats[row.Season][team_ids[row.Wteam]]
        lrating = adj_stats[row.Season][team_ids[row.Lteam]]
        if row.Wteam < row.Lteam:
            ratings_pred.append(numpy.hstack((wrating, lrating)))
        else:
            ratings_pred.append(numpy.hstack((lrating, wrating)))
    return numpy.array(ratings), numpy.array(ratings_pred)

# http://netprophetblog.blogspot.com/2012/02/continued-slow-pursuit-of-statistical.html
def statistics(X, X_pred, stat_F):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    compiled_stats = []
    for row in X.itertuples(index=False):
        stats[row.Season][row.Wteam]['score'].append(row.Wscore)
        stats[row.Season][row.Wteam]['score-against'].append(row.Lscore)
        stats[row.Season][row.Lteam]['score'].append(row.Lscore)
        stats[row.Season][row.Lteam]['score-against'].append(row.Wscore)
        stats[row.Season][row.Wteam]['fgm'].append(row.Wfgm)
        stats[row.Season][row.Wteam]['fgm-against'].append(row.Lfgm)
        stats[row.Season][row.Lteam]['fgm'].append(row.Lfgm)
        stats[row.Season][row.Lteam]['fgm-against'].append(row.Wfgm)
        stats[row.Season][row.Wteam]['fga'].append(row.Wfga)
        stats[row.Season][row.Wteam]['fga-against'].append(row.Lfga)
        stats[row.Season][row.Lteam]['fga'].append(row.Lfga)
        stats[row.Season][row.Lteam]['fga-against'].append(row.Wfga)
        stats[row.Season][row.Wteam]['fgm3'].append(row.Wfgm3)
        stats[row.Season][row.Wteam]['fgm3-against'].append(row.Lfgm3)
        stats[row.Season][row.Lteam]['fgm3'].append(row.Lfgm3)
        stats[row.Season][row.Lteam]['fgm3-against'].append(row.Wfgm3)
        stats[row.Season][row.Wteam]['fga3'].append(row.Wfga3)
        stats[row.Season][row.Wteam]['fga3-against'].append(row.Lfga3)
        stats[row.Season][row.Lteam]['fga3'].append(row.Lfga3)
        stats[row.Season][row.Lteam]['fga3-against'].append(row.Wfga3)
        stats[row.Season][row.Wteam]['ftm'].append(row.Wftm)
        stats[row.Season][row.Wteam]['ftm-against'].append(row.Lftm)
        stats[row.Season][row.Lteam]['ftm'].append(row.Lftm)
        stats[row.Season][row.Lteam]['ftm-against'].append(row.Wftm)
        stats[row.Season][row.Wteam]['fta'].append(row.Wfta)
        stats[row.Season][row.Wteam]['fta-against'].append(row.Lfta)
        stats[row.Season][row.Lteam]['fta'].append(row.Lfta)
        stats[row.Season][row.Lteam]['fta-against'].append(row.Wfta)
        stats[row.Season][row.Wteam]['or'].append(row.Wor)
        stats[row.Season][row.Wteam]['or-against'].append(row.Lor)
        stats[row.Season][row.Lteam]['or'].append(row.Lor)
        stats[row.Season][row.Lteam]['or-against'].append(row.Wor)
        stats[row.Season][row.Wteam]['dr'].append(row.Wdr)
        stats[row.Season][row.Wteam]['dr-against'].append(row.Ldr)
        stats[row.Season][row.Lteam]['dr'].append(row.Ldr)
        stats[row.Season][row.Lteam]['dr-against'].append(row.Wdr)
        stats[row.Season][row.Wteam]['ast'].append(row.Wast)
        stats[row.Season][row.Wteam]['ast-against'].append(row.Last)
        stats[row.Season][row.Lteam]['ast'].append(row.Last)
        stats[row.Season][row.Lteam]['ast-against'].append(row.Wast)
        stats[row.Season][row.Wteam]['to'].append(row.Wto)
        stats[row.Season][row.Wteam]['to-against'].append(row.Lto)
        stats[row.Season][row.Lteam]['to'].append(row.Lto)
        stats[row.Season][row.Lteam]['to-against'].append(row.Wto)
        stats[row.Season][row.Wteam]['stl'].append(row.Wstl)
        stats[row.Season][row.Wteam]['stl-against'].append(row.Lstl)
        stats[row.Season][row.Lteam]['stl'].append(row.Lstl)
        stats[row.Season][row.Lteam]['stl-against'].append(row.Wstl)
        stats[row.Season][row.Wteam]['blk'].append(row.Wblk)
        stats[row.Season][row.Wteam]['blk-against'].append(row.Lblk)
        stats[row.Season][row.Lteam]['blk'].append(row.Lblk)
        stats[row.Season][row.Lteam]['blk-against'].append(row.Wblk)
        stats[row.Season][row.Wteam]['pf'].append(row.Wpf)
        stats[row.Season][row.Wteam]['pf-against'].append(row.Lpf)
        stats[row.Season][row.Lteam]['pf'].append(row.Lpf)
        stats[row.Season][row.Lteam]['pf-against'].append(row.Wpf)
        wstats = stat_F(stats[row.Season][row.Wteam])
        lstats = stat_F(stats[row.Season][row.Lteam])
        if not wstats:
            wstats = [0] * int(len(compiled_stats[0]) / 2)
        if not lstats:
            lstats = [0] * int(len(compiled_stats[0]) / 2)
        if row.Wteam < row.Lteam:
            compiled_stats.append(numpy.hstack((wstats, lstats)))
        else:
            compiled_stats.append(numpy.hstack((lstats, wstats)))
    compiled_stats_pred = []
    for row in X_pred.itertuples(index=False):
        wstats = stat_F(stats[row.Season][row.Wteam])
        lstats = stat_F(stats[row.Season][row.Lteam])
        if row.Wteam < row.Lteam:
            compiled_stats_pred.append(numpy.hstack((wstats, lstats)))
        else:
            compiled_stats_pred.append(numpy.hstack((lstats, wstats)))
    return numpy.array(compiled_stats), numpy.array(compiled_stats_pred)

def assemble_features(pre_X, X, X_pred):
    # strength of schedule
    rpi1, rpi1_predict = modified_rpi(X, X_pred, weights=(.15, .15, .7))
    rpi2, rpi2_predict = modified_rpi(X, X_pred, weights=(.25, .25, .5))
    rpi3, rpi3_predict = modified_rpi(X, X_pred, weights=(.25, .5, .25))
    # expected wins
    pythag1, pythag1_predict = pythagorean_expectation(X, X_pred, exponent=10.25)
    pythag2, pythag2_predict = pythagorean_expectation(X, X_pred, exponent=13.91)
    pythag3, pythag3_predict = pythagorean_expectation(X, X_pred, exponent=16.5)
    # ratings
    markov, markov_predict = advanced_statistic_ratings(X, X_pred, markov_stats, pre_X)
    offdef, offdef_predict = advanced_statistic_ratings(X, X_pred, adjust_stats, pre_X)
    # stats
    descript_stats, descript_stats_predict = statistics(X, X_pred, describe_stats)
    derived_stats, derived_stats_predict = statistics(X, X_pred, derive_stats)

    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(rpi1, columns=['rpi_1', 'rpi_2'])], axis=1)
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(rpi2, columns=['rpi_3', 'rpi_4'])], axis=1)
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(rpi3, columns=['rpi_5', 'rpi_6'])], axis=1)
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(pythag1, columns=['pythag_1', 'pythag_2'])], axis=1)
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(pythag2, columns=['pythag_3', 'pythag_4'])], axis=1)
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(pythag3, columns=['pythag_5', 'pythag_6'])], axis=1)
    markov_cols = ['markov_' + str(i) for i in range(20)]
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(markov, columns=markov_cols)], axis=1)
    offdef_cols = ['offdef_' + str(i) for i in range(40)]
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(offdef, columns=offdef_cols)], axis=1)
    descript_stat_cols = ['stat_a' + str(i) for i in range(280)] # 56 x 5
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(descript_stats, columns=descript_stat_cols)], axis=1)
    derive_stat_cols = ['stat_b' + str(i) for i in range(756)] # 28 x 27
    X = pandas.concat([X.reset_index(drop=True), pandas.DataFrame(derived_stats, columns=derive_stat_cols)], axis=1)

    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(rpi1_predict, columns=['rpi_a', 'rpi_b'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(rpi2_predict, columns=['rpi_c', 'rpi_d'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(rpi3_predict, columns=['rpi_e', 'rpi_f'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True),
                            pandas.DataFrame(pythag1_predict, columns=['pythag_a', 'pythag_b'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True),
                            pandas.DataFrame(pythag2_predict, columns=['pythag_c', 'pythag_d'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True),
                            pandas.DataFrame(pythag3_predict, columns=['pythag_e', 'pythag_f'])], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(markov_predict, columns=markov_cols)], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(offdef_predict, columns=offdef_cols)], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(descript_stats_predict, columns=descript_stat_cols)], axis=1)
    X_pred = pandas.concat([X_pred.reset_index(drop=True), pandas.DataFrame(derived_stats_predict, columns=derive_stat_cols)], axis=1)

    return X, X_pred.values.astype('float64')
