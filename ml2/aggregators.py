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


from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd


TOURNEY_START_DAY = 134


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

def modified_rpi(X, weights=(.15, .15, .7)):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    rpis = []
    for row in X.itertuples(index=False):
        if row.Daynum < TOURNEY_START_DAY:
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
    return np.array(rpis)

def _descriptive_stats(results):
    described = []
    for stat in results.keys():
        described.append(min(results[stat]))
        described.append(max(results[stat]))
        described.append(np.std(results[stat]))
        described.append(np.median(results[stat]))
        described.append(np.mean(results[stat]))
    return described

def statistics(X):
    stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    compiled_stats = []
    for row in X.itertuples(index=False):
        if row.Daynum < TOURNEY_START_DAY:
            wposs = row.Wfga - row.Wor + row.Wto + .475 * row.Wfta
            lposs = row.Lfga - row.Lor + row.Lto + .475 * row.Lfta
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
        wstats = _descriptive_stats(stats[row.Season][row.Wteam])
        lstats = _descriptive_stats(stats[row.Season][row.Lteam])
        if not wstats:
            wstats = [0] * int(len(compiled_stats[0]) / 2)
        if not lstats:
            lstats = [0] * int(len(compiled_stats[0]) / 2)
        if row.Wteam < row.Lteam:
            compiled_stats.append(np.hstack((wstats, lstats)))
        else:
            compiled_stats.append(np.hstack((lstats, wstats)))
    return np.array(compiled_stats)

def _all_teams(data):
    seasons = data['Season'].unique()
    teams = {}
    for season in seasons:
        season_games = data.query('Season == %s' % season)
        sorted_teams = sorted((season_games['Wteam'].append(season_games['Lteam'])).unique())
        teams[season] = {X:i for i, X in enumerate(sorted_teams)}
    return teams

def custom_ratings(X, rating_F, preseason_games=None):
    first_day = min(np.unique(X['Daynum']))
    if preseason_games is not None:
        X = pd.concat([preseason_games, X], ignore_index=True)
    teams = _all_teams(X)
    seasons = np.unique(X['Season'])
    stat_categories = {X:i for i, X in enumerate(['points'])}
    stats = {season: np.zeros((len(teams[season]), len(teams[season]), len(stat_categories))) for season in seasons}
    ratings = []
    day = 0
    adj_stats = None
    for row in X.itertuples(index=False):
        wteam_id = teams[row.Season][row.Wteam]
        lteam_id = teams[row.Season][row.Lteam]
        stats[row.Season][wteam_id][lteam_id][stat_categories['points']] += row.Wscore
        stats[row.Season][lteam_id][wteam_id][stat_categories['points']] += row.Lscore
        if row.Daynum >= first_day:
            if row.Daynum > day or row.Daynum == 0:
                adj_stats = rating_F(stats)
                day = row.Daynum
            team_ids = teams[row.Season]
            wrating = adj_stats[row.Season][team_ids[row.Wteam]]
            lrating = adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                ratings.append(np.hstack((wrating, lrating)))
            else:
                ratings.append(np.hstack((lrating, wrating)))
    return np.array(ratings)
