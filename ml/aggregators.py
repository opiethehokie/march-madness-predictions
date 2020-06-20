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


from collections import defaultdict
from functools import partial

import itertools
import numpy as np
import scipy

from statsmodels.tsa.api import SimpleExpSmoothing, Holt


TOURNEY_START_DAY = 134


# these are calculated using stats from the game to be predicted which seems wrong, but it handles the first
# game problem and we don't use any tourney stats, so I think it's ok

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

def modified_rpi(X, start_day, weights=(.15, .15, .7)):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    rpis = []
    for row in X.itertuples(index=False):
        if row.Daynum >= start_day:
            wrpi = _rpi(stats[row.Season], row.Wteam, weights)
            lrpi = _rpi(stats[row.Season], row.Lteam, weights)
            if row.Wteam < row.Lteam:
                rpis.append(wrpi - lrpi)
            else:
                rpis.append(lrpi - wrpi)
        if row.Daynum < TOURNEY_START_DAY:
            stats[row.Season][row.Wteam]['opponents'].append(row.Lteam)
            stats[row.Season][row.Lteam]['opponents'].append(row.Wteam)
            stats[row.Season][row.Wteam]['results'].append(1)
            stats[row.Season][row.Lteam]['results'].append(0)
    return np.array(rpis)

def descriptive_stats(results):
    described = []
    for stat in results.keys():
        described.append(min(results[stat]))
        described.append(max(results[stat]))
        described.append(np.median(results[stat]))
        described.append(np.mean(results[stat]))
        described.append(np.var(results[stat])) # second moment
        described.append(scipy.stats.skew(results[stat])) #third moment
        described.append(scipy.stats.kurtosis(results[stat])) # fourth moment
    return described

def time_series_stats(results):
    timed = []
    for stat in results.keys():
        timed.append(np.mean(results[stat][-1]))
        timed.append(np.mean(results[stat][-5:])) # simple 5 game moving average
        mean = np.mean(results[stat])
        timed.append(len(list(itertools.takewhile(lambda x, mean=mean: x > mean, reversed(results[stat]))))) # monotonicity
        timed.append(len(list(itertools.takewhile(lambda x, mean=mean: x < mean, reversed(results[stat])))))
        if len(results[stat]) > 1:
            timed.append(SimpleExpSmoothing(results[stat]).fit(smoothing_level=.3, optimized=False).fittedvalues[-1]) # exponential smoothing
            timed.append(Holt(results[stat]).fit(smoothing_level=.5, smoothing_slope=.5, optimized=False).fittedvalues[-1]) # Holt's linear trend
        else:
            timed.append(results[stat][0])
            timed.append(results[stat][0])
    return timed

def vanilla_stats(results):
    stats = []
    for stat in results.keys():
        stats.append(np.mean(results[stat]))
    return stats

def statistics(X, start_day, stat_F):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    compiled_stats = []
    for row in X.itertuples(index=False):
        if row.Daynum >= start_day:
            wstats = stat_F(stats[row.Season][row.Wteam])
            lstats = stat_F(stats[row.Season][row.Lteam])
            if row.Wteam < row.Lteam:
                compiled_stats.append(np.subtract(wstats, lstats))
            else:
                compiled_stats.append(np.subtract(lstats, wstats))
        if row.Daynum < TOURNEY_START_DAY:
            wposs = row.Wfga - row.Wor + row.Wto + .475 * row.Wfta
            lposs = row.Lfga - row.Lor + row.Lto + .475 * row.Lfta
            # interaction features based on commonly used metrics
            stats[row.Season][row.Wteam]['eff_field_goal_percent'].append((row.Wfgm + .5 * row.Wfgm3) / row.Wfga)
            stats[row.Season][row.Lteam]['eff_field_goal_percent'].append((row.Lfgm + .5 * row.Lfgm3) / row.Lfga)
            stats[row.Season][row.Wteam]['true_shooting'].append(.5 * row.Wscore / (row.Wfga + 0.475 * row.Wfta))
            stats[row.Season][row.Lteam]['true_shooting'].append(.5 * row.Lscore / (row.Lfga + 0.475 * row.Lfta))
            stats[row.Season][row.Wteam]['rebound_rate'].append(row.Wor / (row.Wor + row.Ldr))
            stats[row.Season][row.Lteam]['rebound_rate'].append(row.Lor / (row.Lor + row.Wdr))
            stats[row.Season][row.Wteam]['free_throw_rate'].append((row.Wfta / row.Wftm) if row.Wftm > 0 else 0)
            stats[row.Season][row.Lteam]['free_throw_rate'].append((row.Lfta / row.Lftm) if row.Lftm > 0 else 0)
            stats[row.Season][row.Wteam]['turnover_rate'].append(row.Wto / wposs)
            stats[row.Season][row.Lteam]['turnover_rate'].append(row.Lto / lposs)
            stats[row.Season][row.Wteam]['assist_rate'].append(row.Wast / row.Wfgm)
            stats[row.Season][row.Lteam]['assist_rate'].append(row.Last / row.Lfgm)
            stats[row.Season][row.Wteam]['block_rate'].append(row.Wblk / row.Lfga)
            stats[row.Season][row.Lteam]['block_rate'].append(row.Lblk / row.Wfga)
            stats[row.Season][row.Wteam]['steal_rate'].append(row.Wstl / lposs)
            stats[row.Season][row.Lteam]['steal_rate'].append(row.Lstl / wposs)
            stats[row.Season][row.Wteam]['score_rate'].append(row.Wscore / wposs)
            stats[row.Season][row.Lteam]['score_rate'].append(row.Lscore / lposs)
            stats[row.Season][row.Wteam]['foul_rate'].append(row.Wpf / lposs)
            stats[row.Season][row.Lteam]['foul_rate'].append(row.Lpf / wposs)
    return np.array(compiled_stats)

def _all_teams(data):
    seasons = data['Season'].unique()
    teams = {}
    for season in seasons:
        season_games = data.query('Season == %s' % season)
        sorted_teams = sorted((season_games['Wteam'].append(season_games['Lteam'])).unique())
        teams[season] = {X:i for i, X in enumerate(sorted_teams)}
    return teams

def custom_ratings(X, start_day, rating_F):
    teams = _all_teams(X)
    seasons = np.unique(X['Season'])
    stat_categories = {X:i for i, X in enumerate(['points'])}
    stats = {season: np.zeros((len(teams[season]), len(teams[season]), len(stat_categories))) for season in seasons}
    ratings = []
    day = start_day - 1
    adj_stats = None
    for row in X.itertuples(index=False):
        if row.Daynum < TOURNEY_START_DAY:
            if row.Daynum > day:
                adj_stats = rating_F(stats)
                day = row.Daynum
            wteam_id = teams[row.Season][row.Wteam]
            lteam_id = teams[row.Season][row.Lteam]
            stats[row.Season][wteam_id][lteam_id][stat_categories['points']] += row.Wscore
            stats[row.Season][lteam_id][wteam_id][stat_categories['points']] += row.Lscore
        if row.Daynum >= start_day:
            team_ids = teams[row.Season]
            wrating = adj_stats[row.Season][team_ids[row.Wteam]]
            lrating = adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                ratings.append(np.subtract(wrating, lrating))
            else:
                ratings.append(np.subtract(lrating, wrating))
    return np.array(ratings)

# https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings

def _elo_pred(elo1, elo2):
    return 1 / (10 ** (-(elo1 - elo2) / 400) + 1)

def _elo_expected_margin(elo_diff):
    return 7.5 + 0.006 * elo_diff

def _elo_update(welo, lelo, mov):
    elo_diff = welo - lelo
    pred = _elo_pred(welo, lelo)
    mult = ((mov + 3) ** 0.8) / _elo_expected_margin(elo_diff)
    update = 20 * mult * (1 - pred)
    return update

def elo(X, start_day):
    stats = defaultdict(partial(defaultdict, lambda: 1500))
    elos = []
    for row in X.itertuples(index=False):
        welo = stats[row.Season][row.Wteam]
        lelo = stats[row.Season][row.Lteam]
        if row.Daynum >= start_day:
            if row.Wteam < row.Lteam:
                elos.append(welo - lelo)
            else:
                elos.append(lelo - welo)
        if row.Daynum < TOURNEY_START_DAY:
            mov = row.Wscore - row.Lscore
            wadvantage = 100 if row.Wloc == 'H' else 0
            ladvantage = 100 if row.Wloc == 'A' else 0
            update = _elo_update(welo + wadvantage, lelo + ladvantage, mov)
            stats[row.Season][row.Wteam] += update
            stats[row.Season][row.Lteam] -= update
    return elos
