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


import os
import time

from tempfile import mkdtemp

import numpy
import pandas

from sklearn.externals import joblib
from sklearn.metrics import log_loss

from ml.classification import train_model
from ml.wrangling import custom_train_test_split


def test_model():
    year = 2013
    sday = 60
    data = pandas.concat([pandas.read_csv('data/regular_season_detailed_results_2016.csv'),
                          pandas.read_csv('data/tourney_detailed_results_2016.csv')]).sort_values(by='Daynum')
    games = (data.pipe(lambda df: df[df.Season >= year])
             .pipe(lambda df: df[df.Season <= year])
             .pipe(lambda df: df[df.Daynum >= sday]))
    X_train, X_test, y_train, y_test = custom_train_test_split(games, year)
    start_time = time.time()
    model = train_model(None, X_train, X_test, y_train, y_test)
    end_time = time.time()
    assert end_time - start_time < 130

    assert model.best_score_ >= -0.59
    assert log_loss(y_test, model.predict_proba(X_test)) <= 0.72

    file = os.path.join(mkdtemp(), 'test.pkl')
    joblib.dump(model, file)
    persisted_model = joblib.load(file)

    fake_boxscores = [[year, 137, 1361, 1328]]
    X_predict = pandas.DataFrame(fake_boxscores, columns=['Season', 'Daynum', 'Wteam', 'Lteam'])
    assert numpy.array_equal(persisted_model.predict_proba(X_predict), model.predict_proba(X_predict))
