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

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

from ml.regression_stacking_cv_classifier import RegressionStackingCVClassifier
from ml.transformers import ColumnSelector, SkewnessTransformer
from ml.wrangling import TOURNEY_START_DAY
from ml.util import print_models


RPI_START = 4
RPI_END = 9

PYTHAG_START = 10
PYTHAG_END = 15

MARKOV_RATING_START = 16
MARKOV_RATING_END = 25

OFFDEF_RATING_START = 26
OFFDEF_RATING_END = 35

DESCRIPT_STAT_START = 36
DESCRIPT_STAT_END = 315

DERIVE_STAT_START = 316
DERIVE_STAT_END = 3395

#TODO http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

def rpi_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, RPI_END + 1)]),
                         StandardScaler(),
                         LinearRegression())

def pythag_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(PYTHAG_START, PYTHAG_END + 1)]),
                         StandardScaler(),
                         LinearRegression())

def markov_rating_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         StandardScaler(),
                         LinearRegression())

def off_def_rating_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         StandardScaler(),
                         LinearRegression())

def descriptive_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DESCRIPT_STAT_START, DESCRIPT_STAT_END + 1)]),
                         StandardScaler(),
                         LinearRegression())

def derived_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DERIVE_STAT_START, DERIVE_STAT_END+1)]),
                         StandardScaler(),
                         LinearRegression())

def mov_to_win(label):
    return int(label > 0)


@print_models
def train_model(X_train, y_train, random_state=None, n_jobs=1, regressors=None):

    if not regressors:
        regressors = [pythag_regression(), rpi_regression()]

    stacker = make_pipeline(SelectKBest(score_func=f_regression),
                            RegressionStackingCVClassifier(regressors=regressors,
                                                           meta_classifier=LogisticRegression(),
                                                           to_class_func=mov_to_win))

    # http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
    grid = { #'regressionstackingcvclassifier__pipeline__lasso__alpha': [1, 10, 100],
             #'regressionstackingcvclassifier__pipeline__lasso__normalize': [True, False],
             #'regressionstackingcvclassifier__pipeline__ridge__alpha': [1, 10, 100],
             #'regressionstackingcvclassifier__pipeline__ridge__fit_intercept': [True, False],
             #'regressionstackingcvclassifier__pipeline__ridge__normalize': [True, False],
             'selectkbest__k': ['all'],
             #'regressionstackingcvclassifier__pipeline__meta-logisticregression__C': [.01, .1, 1],
             #'regressionstackingcvclassifier__pipeline__meta-logisticregression__penalty': ['l1', 'l2']
           }
    #print(stacker.get_params().keys())

    cv = custom_cv(X_train)

    scoring = make_scorer(custom_log_loss, needs_proba=True, to_class_func=mov_to_win)

    model = GridSearchCV(estimator=stacker, param_grid=grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model

def custom_cv(X):
    season_col = X[:, 0]
    seasons = numpy.unique(season_col)
    day_col = X[:, 1]
    return [(numpy.where((season_col != season) | (day_col < TOURNEY_START_DAY))[0],
             numpy.where((season_col == season) & (day_col >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]

def custom_log_loss(y_true, y_pred, to_class_func):
    y_true = numpy.fromiter((to_class_func(yi) for yi in y_true), y_true.dtype)
    return log_loss(y_true, y_pred, labels=[0, 1])
