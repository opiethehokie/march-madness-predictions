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


import numpy
import pandas
import pytest

from ml import wrangling


@pytest.fixture
def box_scores():
    headers = ['Season', 'Daynum', 'Numot', 'Wloc', 'Wteam', 'Wscore', 'Wfgm', 'Wfga', 'Wfgm3', 'Wfga3',
               'Wftm', 'Wfta', 'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf', 'Lteam', 'Lscore',
               'Lfgm', 'Lfga', 'Lfgm3', 'Lfga3', 'Lftm', 'Lfta', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf']
    box_score1 = [2010, 1, 0, 'H', 4, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                  6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    box_score2 = [2010, 1, 1, 'A', 6, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                  11, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    box_score3 = [2010, 1, 2, 'N', 11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                  4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    return pandas.DataFrame([box_score1, box_score2, box_score3], columns=headers)

def test_dervied_stats():
    season_stats = {'score': [5, 5], 'score-against': [5, 2],
                    'fgm': [5, 0], 'fgm-against': [5, 3],
                    'fga': [5, 77], 'fga-against': [5, 3]}
    assert len(wrangling.derive_stats(season_stats)) == 15

def test_descriptive_stats():
    season_stats = {'score': [5, 5]}
    stats = wrangling.describe_stats(season_stats)
    assert len(stats) == 5
    assert stats == [5, 5, 0.0, 5.0, 5.0]

def test_modified_rpi():
    season_stats = {1: {'opponents': [2, 3], 'results': [1, 1]}, 2: {'opponents': [1, 3], 'results': [0, 0]},
                    3: {'opponents': [1, 2], 'results': [0, 1]}}
    assert wrangling._opponents_win_percent(season_stats, [2, 3]) == .25
    assert wrangling._opponents_opponents_win_percent(season_stats, [2, 3]) == .625
    assert wrangling._rpi(season_stats, 1, [.25, .5, .25]) == .25 * 1 + .5 * .25 + .25 * .625

#pylint: disable=redefined-outer-name
def test_overtime_adjustments(box_scores):
    assert numpy.array_equal(numpy.array([[2010, 1, 0, 'H', 4, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                                          [2010, 1, 1, 'A', 6, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
                                           11, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88],
                                          [2010, 1, 2, 'N', 11, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                           4, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]], dtype=object),
                             wrangling.adjust_overtime_games(box_scores).values)

#pylint: disable=redefined-outer-name
def test_create_synthetic_games(box_scores):
    assert numpy.array_equal(numpy.array([[2010, 1, 0, 'H', 4, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
                                           6, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
                                          [2010, 1, 1, 'A', 6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                           11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                          [2010, 1, 2, 'N', 11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                                          [2010, 1, 2, 'N', 11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                                          [2010, 1, 2, 'N', 11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]], dtype=object),
                             wrangling.create_synthetic_games(box_scores, home_factor=.5, neutral_site_factor=2).values)
