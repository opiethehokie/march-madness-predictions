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

from ml.predictions import train_regressor, custom_cv
from ml.transformers import ColumnSelector
from ml.util import mov_to_win_percent
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
    assert X_train.shape == (9789, 6)
    assert X_test.shape == (67, 6)
    assert y_train.shape == (9789,)
    assert y_test.shape == (67,)
    assert isinstance(X_train, numpy.ndarray)
    assert isinstance(X_test, numpy.ndarray)
    assert isinstance(y_train, numpy.ndarray)
    assert isinstance(y_test, numpy.ndarray)

    model = train_regressor(X_train, y_train, regressors=[make_pipeline(ColumnSelector(cols=[4, 5]), StandardScaler(), LinearRegression())])
    y_predict = model.predict(X_test)
    y_predict_probas = [mov_to_win_percent(yi) for yi in y_predict]
    assert log_loss(y_test, y_predict_probas) <= 0.69

    file = os.path.join(mkdtemp(), 'test.pkl')
    joblib.dump(model, file)
    persisted_model = joblib.load(file)
    fake_boxscores = [[year, 137, 1361, 1328, .55, .52]]
    X_predict = pandas.DataFrame(fake_boxscores)
    assert numpy.array_equal(persisted_model.predict(X_predict), model.predict(X_predict))

def test_for_leakage():
    year = 2017
    sday = 60

    data1 = pandas.concat([pandas.read_csv('data/regular_season_detailed_results_2018.csv'),
                           pandas.read_csv('data/tourney_detailed_results_2017.csv')]).sort_values(by=['Daynum', 'Wteam', 'Lteam'])
    games1 = (data1.pipe(lambda df: df[df.Season >= year - 1])
              .pipe(lambda df: df[df.Season <= year])
              .pipe(lambda df: df[df.Daynum >= sday]))
    rpis1, _ = modified_rpi(games1, pandas.DataFrame([]))
    games1 = pandas.concat([games1.reset_index(drop=True), pandas.DataFrame(rpis1, columns=['rpi1', 'rpi2'])], axis=1)
    X_train1, _, y_train1, _ = custom_train_test_split(games1, year)
    model1 = train_regressor(X_train1, y_train1, regressors=[make_pipeline(ColumnSelector(cols=[4, 5]), StandardScaler(), LinearRegression())])

    data2 = pandas.concat([pandas.read_csv('data/regular_season_detailed_results_2018.csv'),
                           pandas.read_csv('data/tourney_detailed_results_2016.csv')]).sort_values(by=['Daynum', 'Wteam', 'Lteam'])
    games2 = (data2.pipe(lambda df: df[df.Season >= year - 1])
              .pipe(lambda df: df[df.Season <= year])
              .pipe(lambda df: df[df.Daynum >= sday]))
    rpis2, _ = modified_rpi(games2, pandas.DataFrame([]))
    games2 = pandas.concat([games2.reset_index(drop=True), pandas.DataFrame(rpis2, columns=['rpi1', 'rpi2'])], axis=1)
    X_train2, _, y_train2, _ = custom_train_test_split(games2, year)
    model2 = train_regressor(X_train2, y_train2, regressors=[make_pipeline(ColumnSelector(cols=[4, 5]), StandardScaler(), LinearRegression())])

    assert X_train1.shape == X_train2.shape
    assert numpy.array_equal(X_train1, X_train2)
    assert numpy.array_equal(y_train1, y_train2)
    numpy.testing.assert_almost_equal(model1.best_score_, model2.best_score_, 1e-4)

def test_custom_cv():
    X = numpy.array([[2013, 50],
                     [2013, 140],
                     [2014, 55],
                     [2014, 138],
                     [2015, 22]
                    ])
    indices = custom_cv(X)
    assert len(indices) == 2
    assert numpy.array_equal(numpy.array([2, 4], dtype=numpy.int64), indices[0][0])
    assert numpy.array_equal(numpy.array([1], dtype=numpy.int64), indices[0][1])
    assert numpy.array_equal(numpy.array([0, 4], dtype=numpy.int64), indices[1][0])
    assert numpy.array_equal(numpy.array([3], dtype=numpy.int64), indices[1][1])
