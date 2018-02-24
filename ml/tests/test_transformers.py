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

from ml.transformers import (HomeCourtTransformer, ModifiedRPITransformer, OvertimeTransformer, SkewnessTransformer)


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

def test_rpi_transformer():
    season_stats = {1: {'opponents': [2, 3], 'results': [1, 1]}, 2: {'opponents': [1, 3], 'results': [0, 0]},
                    3: {'opponents': [1, 2], 'results': [0, 1]}}
    rpi = ModifiedRPITransformer(weights=[.25, .5, .25])
    assert rpi._opponents_win_percent(season_stats, [2, 3]) == .25
    assert rpi._opponents_opponents_win_percent(season_stats, [2, 3]) == .625
    assert rpi._rpi(season_stats, 1) == .25 * 1 + .5 * .25 + .25 * .625

def test_skew_transformer():
    data = numpy.array([[9, 9, 1],
                        [8, 9, 2],
                        [1, 1, 1]])
    skew = SkewnessTransformer(max_skew=.7, lmbda=.5)
    assert not numpy.array_equal(skew.transform(data), data)
    skew = SkewnessTransformer(max_skew=.5, lmbda=0)
    assert not numpy.array_equal(skew.transform(data), data)
    skew = SkewnessTransformer(max_skew=.5, lmbda=None)
    assert not numpy.array_equal(skew.transform(data), data)
    skew = SkewnessTransformer(max_skew=100, lmbda=None)
    assert numpy.array_equal(skew.transform(data), data)

#pylint: disable=redefined-outer-name
def test_overtime_transformer(box_scores):
    assert numpy.array_equal(numpy.array([[2010, 1, 0, 'H', 4, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                                          [2010, 1, 1, 'A', 6, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
                                           11, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88],
                                          [2010, 1, 2, 'N', 11, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                           4, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]], dtype=object),
                             OvertimeTransformer().fit_transform(box_scores).as_matrix())

#pylint: disable=redefined-outer-name
def test_home_court_transformer(box_scores):
    assert numpy.array_equal(numpy.array([[2010, 1, 0, 'H', 4, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
                                           6, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
                                          [2010, 1, 1, 'A', 6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                           11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                          [2010, 1, 2, 'N', 11, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                           4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]], dtype=object),
                             HomeCourtTransformer(factor=.5).fit_transform(box_scores).as_matrix())
