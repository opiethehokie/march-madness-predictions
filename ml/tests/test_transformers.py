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


import numpy
import pandas

from ml.transformers import HomeCourtTransformer, ModifiedRPITransformer, MovStdDevTransformer, RatingTransformer


def test_home_court_transformer():
    headers = ['Wteam', 'Lteam', 'Wloc', 'Daynum']
    box_score1 = [1, 2, 'H', 1]
    box_score2 = [1, 2, 'A', 2]
    box_score3 = [1, 2, 'N', 3]
    box_score4 = [2, 1, 'H', 4]
    box_score5 = [2, 1, 'A', 5]
    box_score6 = [2, 1, 'N', 6]
    X = pandas.DataFrame([box_score1, box_score2, box_score3, box_score4, box_score5, box_score6], columns=headers)
    assert numpy.array_equal(numpy.array([[1, 0], [0, 1], [0, 0], [0, 1], [1, 0], [0, 0]]),
                             HomeCourtTransformer().fit_transform(X))

def test_consistency_transformer():
    headers = ['Season', 'Wteam', 'Lteam', 'Wscore', 'Lscore', 'Daynum']
    box_score1 = [1, 1, 2, 60, 50, 4]
    box_score2 = [1, 2, 1, 55, 50, 5]
    box_score3 = [2, 1, 2, 66, 64, 6]
    X = pandas.DataFrame([box_score1, box_score2, box_score3], columns=headers)
    assert numpy.array_equal(numpy.array([[0, 0], [7.5, 7.5], [0, 0]]),
                             MovStdDevTransformer(25).fit_transform(X))
    assert numpy.array_equal(numpy.array([[0, 0], [5, 5], [0, 0]]),
                             MovStdDevTransformer(max_mov=5).fit_transform(X))

def test_rpi_transformer():
    season_stats = {1: {'opponents': [2, 3], 'results': [1, 1]}, 2: {'opponents': [1, 3], 'results': [0, 0]},
                    3: {'opponents': [1, 2], 'results': [0, 1]}}
    rpi = ModifiedRPITransformer(weights=[.25, .5, .25])
    assert rpi._opponents_win_percent(season_stats, [2, 3]) == .25
    assert rpi._opponents_opponents_win_percent(season_stats, [2, 3]) == .625
    assert rpi._rpi(season_stats, 1) == .25 * 1 + .5 * .25 + .25 * .625

def fake_games():
    headers = ['Season', 'Daynum', 'Wteam', 'Wscore', 'Wfgm', 'Wfga', 'Wfgm3', 'Wfga3', 'Wftm', 'Wfta',
               'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf', 'Lteam', 'Lscore', 'Lfgm', 'Lfga',
               'Lfgm3', 'Lfga3', 'Lftm', 'Lfta', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf']
    box_score1 = [2010, 1, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    box_score2 = [2010, 2, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    box_score3 = [2010, 3, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    return pandas.DataFrame([box_score1, box_score2, box_score3], columns=headers)

def test_calculate_stats_all():
    teams = {2010: {4:0, 6:1, 11:2}}
    box_scores = fake_games()
    previous_games = {2010: {4: [box_scores.iloc[0], box_scores.iloc[2]], 6: [box_scores.iloc[1], box_scores.iloc[0]],
                             11: [box_scores.iloc[2], box_scores.iloc[1]]}}
    stats = RatingTransformer(None, pandas.DataFrame())._recalc_stats(teams, previous_games, 2)
    assert numpy.any(stats[2010][0][1])
    assert numpy.any(stats[2010][0][2])
    assert numpy.any(stats[2010][1][0])
    assert numpy.any(stats[2010][1][2])
    assert numpy.any(stats[2010][2][0])
    assert numpy.any(stats[2010][2][1])
    assert not numpy.any(stats[2010][0][0])
    assert not numpy.any(stats[2010][1][1])
    assert not numpy.any(stats[2010][2][2])

def test_calculate_stats_partial():
    teams = {2010: {4:0, 6:1, 11:2}}
    box_scores = fake_games()
    previous_games = {2010: {4: [box_scores.iloc[0], box_scores.iloc[2]], 6: [box_scores.iloc[1], box_scores.iloc[0]],
                             11: [box_scores.iloc[2], box_scores.iloc[1]]}}
    stats = RatingTransformer(None, pandas.DataFrame())._recalc_stats(teams, previous_games, 1)
    assert numpy.any(stats[2010][0][1])
    assert numpy.any(stats[2010][0][2])
    assert numpy.any(stats[2010][1][0])
    assert numpy.any(stats[2010][1][2])
    assert numpy.any(stats[2010][2][0])
    assert numpy.any(stats[2010][2][1])
    assert not numpy.any(stats[2010][0][0])
    assert not numpy.any(stats[2010][1][1])
    assert not numpy.any(stats[2010][2][2])
