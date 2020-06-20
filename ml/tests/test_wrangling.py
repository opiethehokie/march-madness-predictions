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
    return pd.DataFrame([box_score1, box_score2, box_score3], columns=headers)

def test_custom_cv():
    X = np.array([[2013, 50],
                  [2013, 140],
                  [2014, 55],
                  [2014, 138],
                  [2015, 22]])
    X = pd.DataFrame(X)
    X.index = pd.MultiIndex.from_arrays(X[[0, 1]].values.T, names=['Season', 'Daynum'])
    indices = wrangling._custom_cv(X)
    print(indices)
    assert len(indices) == 2
    assert np.array_equal(np.array([0], dtype=np.int64), indices[0][0])
    assert np.array_equal(np.array([1], dtype=np.int64), indices[0][1])
    assert np.array_equal(np.array([2], dtype=np.int64), indices[1][0])
    assert np.array_equal(np.array([3], dtype=np.int64), indices[1][1])
