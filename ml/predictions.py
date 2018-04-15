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
from mlxtend.regressor import StackingCVRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, HuberRegressor, Lars
from sklearn.metrics import r2_score, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from ml.transformers import ColumnSelector, DiffTransformer
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


def rpi_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, RPI_END + 1)]),
                         StandardScaler(),
                         Ridge(random_state=random_state, alpha=1))


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


def markov_rating_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         StandardScaler(),
                         RFE(LinearSVR(random_state=random_state, C=1), n_features_to_select=2))


def off_def_rating_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         DiffTransformer(),
                         StandardScaler(),
                         KNeighborsRegressor(n_neighbors=5))


def descriptive_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DESCRIPT_STAT_START, DESCRIPT_STAT_END + 1)]),
                         DiffTransformer(),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=2),
                         Lars())


def derived_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DERIVE_STAT_START, DERIVE_STAT_END+1)]),
                         RBFSampler(),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=4),
                         SymbolicRegressor(random_state=random_state, stopping_criteria=0.05))


@print_models
def train_regressor(X_train, y_train, regressors=None):

    if not regressors:
        regressors = [rpi_regression1(),
                      rpi_regression2(),
                      rpi_regression3(),
                      pythag_regression(),
                      markov_rating_regression(),
                      off_def_rating_regression(),
                      descriptive_stat_regression(),
                      derived_stat_regression()
                     ]

    grid = {
        #'stackingcvregressor__pipeline-1__ridge__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
        #'stackingcvregressor__pipeline-4__lasso__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
        #'stackingcvregressor__pipeline-5__rfe__n_features_to_select': Integer(2, 8),
        #'stackingcvregressor__pipeline-5__rfe__estimator__C': Real(1e-2, 1e+0, prior='log-uniform'),
        #'stackingcvregressor__pipeline-6__kneighborsregressor__n_neighbors': Categorical([3, 5, 8]),
        #'stackingcvregressor__pipeline-7__pca__n_components': Integer(2, 8),
        #'stackingcvregressor__pipeline-8__pca__n_components': Integer(2, 8),
        'stackingcvregressor__use_features_in_secondary': Categorical([True]),
        #'stackingcvregressor__meta-mlpregressor__hidden_layer_sizes': Categorical([(5,), (7,), (10,), (5, 5), (7, 7), (10, 10)]),
        #'stackingcvregressor__meta-mlpregressor__activation': Categorical(['logistic', 'relu']),
        #'stackingcvregressor__meta-mlpregressor__alpha': Real(1e-3, 1e-1, prior='log-uniform')
    }

    iters = len(grid.keys()) * 5 if len(grid.keys()) > 1 else 1

    stacker = make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, DERIVE_STAT_END + 1)]),
                            StackingCVRegressor(regressors=regressors, shuffle=True, cv=5, use_features_in_secondary=True,
                                                meta_regressor=MLPRegressor(random_state=random_state, max_iter=1000, activation='relu',
                                                                            hidden_layer_sizes=(7, 7), alpha=.001)))

    print('Training model ...')
    t1 = time.time()
    model = BayesSearchCV(stacker, grid, cv=custom_cv(X_train), scoring=make_scorer(r2_score), n_jobs=4,
                          random_state=random_state, n_iter=iters)
    model.fit(X_train, y_train)
    t2 = time.time()
    print('Training took %f seconds' % (t2 - t1))

    return model

def custom_cv(X):
    season_col = X[:, 0]
    seasons = numpy.unique(season_col)
    day_col = X[:, 1]
    return [(numpy.where((season_col != season) & (day_col < TOURNEY_START_DAY))[0],
             numpy.where((season_col == season) & (day_col >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]
