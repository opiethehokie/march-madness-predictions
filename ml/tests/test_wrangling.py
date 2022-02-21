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
