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


import time

import numpy

from gplearn.genetic import SymbolicRegressor
from polylearn import FactorizationMachineRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, RFE, VarianceThreshold
from sklearn.linear_model import (LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression,
                                  BayesianRidge, HuberRegressor, PassiveAggressiveRegressor)
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVR
#from skopt import BayesSearchCV
#from skopt.space import Real, Categorical, Integer

from ml.regression_stacking_cv_classifier import RegressionStackingCVClassifier
from ml.transformers import ColumnSelector, SkewnessTransformer, DiffTransformer
from ml.wrangling import TOURNEY_START_DAY
from ml.util import print_models


RPI_START = 4
RPI_END = 9

PYTHAG_START = 10
PYTHAG_END = 15

MARKOV_RATING_START = 16
MARKOV_RATING_END = 35

OFFDEF_RATING_START = 36
OFFDEF_RATING_END = 75

DESCRIPT_STAT_START = 76
DESCRIPT_STAT_END = 355

DERIVE_STAT_START = 356
DERIVE_STAT_END = 1111

random_state = 42
numpy.random.seed(random_state)

n_jobs = 3


def rpi_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, RPI_END + 1)]),
                         StandardScaler(),
                         Ridge(random_state=random_state, alpha=100))

def rpi_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, RPI_END + 1)]),
                         StandardScaler(),
                         BayesianRidge())

def rpi_regression3():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, RPI_END + 1)]),
                         StandardScaler(),
                         HuberRegressor())

def pythag_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(PYTHAG_START, PYTHAG_END + 1)]),
                         DiffTransformer(),
                         StandardScaler(),
                         Lasso(random_state=random_state, alpha=1))

def markov_rating_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         SkewnessTransformer(lmbda=None),
                         StandardScaler(),
                         RFE(ElasticNet(random_state=random_state, alpha=10), step=.1, n_features_to_select=2))

def markov_rating_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         FactorizationMachineRegressor(n_components=1, fit_linear=False, random_state=random_state))

def off_def_rating_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         StandardScaler(),
                         RFE(ElasticNet(random_state=random_state, alpha=10), step=.05, n_features_to_select=2))

def off_def_rating_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         FeatureAgglomeration(n_clusters=2),
                         LinearRegression())

def descriptive_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DESCRIPT_STAT_START, DESCRIPT_STAT_END + 1)]),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=2),
                         LinearSVR(random_state=random_state, C=.1))

def derived_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DERIVE_STAT_START, DERIVE_STAT_END+1)]),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=4),
                         LinearSVR(random_state=random_state, C=.1))

def mixed_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, OFFDEF_RATING_END + 1)]),
                         StandardScaler(),
                         SelectKBest(k=20),
                         PolynomialFeatures(degree=2),
                         PassiveAggressiveRegressor(tol=.001, max_iter=1000, random_state=random_state))

def mixed_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, OFFDEF_RATING_END + 1)]),
                         VarianceThreshold(threshold=.5),
                         SymbolicRegressor(random_state=random_state, stopping_criteria=0.05))

def mov_to_win(label):
    return int(label > 0)


@print_models
def train_model(X_train, y_train, regressors=None):

    if not regressors:
        regressors = [rpi_regression1(),
                      rpi_regression2(),
                      rpi_regression3(),
                      pythag_regression(),
                      markov_rating_regression1(),
                      markov_rating_regression2(),
                      off_def_rating_regression1(),
                      off_def_rating_regression2(),
                      descriptive_stat_regression(),
                      derived_stat_regression(),
                      mixed_regression1(),
                      mixed_regression2()
                     ]

    grid = {#'meta-logisticregression__C': [.01, .1, 1],
            #'meta-logisticregression__penalty': ['l1', 'l2']
           }

    #grid = {'pipeline-10__pca__n_components': Integer(1, 8),
    #        'meta-logisticregression__C': Real(1e-3, 1e+1, prior='log-uniform'),
    #        'meta-logisticregression__penalty': Categorical(['l1', 'l2'])
    #       }

    stacker = RegressionStackingCVClassifier(regressors=regressors,
                                             meta_classifier=LogisticRegression(penalty='l2', C=1),
                                             to_class_func=mov_to_win)

    print('Training model ...')
    t1 = time.time()
    cv = custom_cv(X_train)
    scoring = make_scorer(custom_log_loss, needs_proba=True, to_class_func=mov_to_win)
    model = GridSearchCV(estimator=stacker, param_grid=grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    #model = BayesSearchCV(stacker, grid, scoring=scoring, cv=cv, n_jobs=n_jobs, random_state=random_state)
    #model.fit(X_train, y_train)
    t2 = time.time()
    print('Training took %f seconds' % (t2 - t1))
    return model

def custom_cv(X):
    season_col = X[:, 0]
    seasons = numpy.unique(season_col)
    day_col = X[:, 1]
    return [(numpy.where((season_col != season) | (day_col < TOURNEY_START_DAY))[0],
             numpy.where((season_col == season) & (day_col >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]

def custom_log_loss(y_true, y_pred, to_class_func):
    y_true = numpy.fromiter((to_class_func(yi) for yi in y_true), y_true.dtype)
    return -1 * log_loss(y_true, y_pred, labels=[0, 1])
