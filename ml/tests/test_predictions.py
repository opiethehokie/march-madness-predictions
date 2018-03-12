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


from tempfile import mkdtemp

import os
import numpy
import pandas

from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.predictions import train_model, custom_cv, custom_log_loss, mov_to_win
from ml.transformers import ColumnSelector
from ml.wrangling import custom_train_test_split, modified_rpi


def test_model():
    year = 2013
    sday = 60
    data = pandas.concat([pandas.read_csv('data/regular_season_detailed_results_2018.csv'),
                          pandas.read_csv('data/tourney_detailed_results_2017.csv')]).sort_values(by='Daynum')

    games = (data.pipe(lambda df: df[df.Season >= year - 2])
             .pipe(lambda df: df[df.Season <= year])
             .pipe(lambda df: df[df.Daynum >= sday]))
    assert games.shape == (9856, 34)

    rpis, _ = modified_rpi(games, pandas.DataFrame([]))
    assert rpis.shape == (9856, 2)

    games = pandas.concat([games.reset_index(drop=True), pandas.DataFrame(rpis, columns=['rpi1', 'rpi2'])], axis=1)
    assert games.shape == (9856, 36)

    X_train, X_test, y_train, y_test = custom_train_test_split(games, year)
    assert X_train.shape == (9793, 6)
    assert X_test.shape == (63, 6)
    assert y_train.shape == (9793,)
    assert y_test.shape == (63,)
    assert isinstance(X_train, numpy.ndarray)
    assert isinstance(X_test, numpy.ndarray)
    assert isinstance(y_train, numpy.ndarray)
    assert isinstance(y_test, numpy.ndarray)

    model = train_model(X_train, y_train, regressors=[make_pipeline(ColumnSelector(cols=[4, 5]), StandardScaler(), LinearRegression())])
    assert log_loss(y_test, model.predict_proba(X_test)) <= 0.65

    file = os.path.join(mkdtemp(), 'test.pkl')
    joblib.dump(model, file)
    persisted_model = joblib.load(file)
    fake_boxscores = [[year, 137, 1361, 1328, .55, .52]]
    X_predict = pandas.DataFrame(fake_boxscores)
    assert numpy.array_equal(persisted_model.predict_proba(X_predict), model.predict_proba(X_predict))

def test_custom_cv():
    X = numpy.array([[2013, 50],
                     [2013, 140],
                     [2014, 55],
                     [2014, 138],
                     [2015, 22]
                    ])
    indices = custom_cv(X)
    assert len(indices) == 2
    assert numpy.array_equal(numpy.array([0, 2, 3, 4], dtype=numpy.int64), indices[0][0])
    assert numpy.array_equal(numpy.array([1], dtype=numpy.int64), indices[0][1])
    assert numpy.array_equal(numpy.array([0, 1, 2, 4], dtype=numpy.int64), indices[1][0])
    assert numpy.array_equal(numpy.array([3], dtype=numpy.int64), indices[1][1])

def test_custom_log_loss():
    movs = numpy.array([4, 5, -3, 11, -22])
    predictions = numpy.array([.6, .6, .48, .8, .2])
    assert custom_log_loss(movs, predictions, mov_to_win) == 0.42437296351341292
