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
from polylearn import FactorizationMachineRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, RFE, VarianceThreshold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVR
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

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
                         Lasso(random_state=random_state, alpha=100))

def markov_rating_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         SkewnessTransformer(lmbda=None),
                         StandardScaler(),
                         RFE(ElasticNet(random_state=random_state, alpha=100), step=.1, n_features_to_select=6))

def markov_rating_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(MARKOV_RATING_START, MARKOV_RATING_END + 1)]),
                         FactorizationMachineRegressor(n_components=2, fit_linear=False, random_state=random_state))

def off_def_rating_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         StandardScaler(),
                         RFE(ElasticNet(random_state=random_state, alpha=100), step=.05, n_features_to_select=2))

def off_def_rating_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(OFFDEF_RATING_START, OFFDEF_RATING_END + 1)]),
                         FeatureAgglomeration(n_clusters=2),
                         LinearRegression())

def descriptive_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DESCRIPT_STAT_START, DESCRIPT_STAT_END + 1)]),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=1),
                         LinearSVR(random_state=random_state, C=.01))

def derived_stat_regression():
    return make_pipeline(ColumnSelector(cols=[i for i in range(DERIVE_STAT_START, DERIVE_STAT_END+1)]),
                         StandardScaler(),
                         PCA(random_state=random_state, n_components=1),
                         LinearSVR(random_state=random_state, C=.01))

def mixed_regression1():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, OFFDEF_RATING_END + 1)]),
                         StandardScaler(),
                         SelectKBest(k=30),
                         PolynomialFeatures(degree=2),
                         PassiveAggressiveRegressor(tol=.001, max_iter=1000, random_state=random_state))

def mixed_regression2():
    return make_pipeline(ColumnSelector(cols=[i for i in range(RPI_START, OFFDEF_RATING_END + 1)]),
                         VarianceThreshold(threshold=.8),
                         SymbolicRegressor(random_state=random_state, stopping_criteria=0.05))

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

    grid = {#'pipeline-1__ridge__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
            #'pipeline-4__lasso__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
            #'pipeline-5__rfe__estimator__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
            #'pipeline-5__rfe__n_features_to_select': Integer(4, 8),
            #'pipeline-6__factorizationmachineregressor__n_components': Integer(1, 2),
            #'pipeline-7__rfe__n_features_to_select': Integer(2, 4),
            #'pipeline-7__rfe__estimator__alpha': Real(1e+0, 1e+2, prior='log-uniform'),
            #'pipeline-8__featureagglomeration__n_clusters': Integer(2, 4),
            #'pipeline-9__pca__n_components': Integer(1, 4),
            #'pipeline-9__linearsvr__C': Real(1e-2, 1e+0, prior='log-uniform'),
            #'pipeline-10__pca__n_components': Integer(1, 4),
            #'pipeline-10__linearsvr__C': Real(1e-2, 1e+0, prior='log-uniform'),
            #'pipeline-11__selectkbest__k': Integer(20, 40),
            #'pipeline-12__variancethreshold__threshold': Real(0, 1),
            'use_features_in_secondary': Categorical([True]),
            #'meta-elasticnet__alpha': Real(1e+0, 1e+2, prior='log-uniform')
           }

    iters = len(grid.keys()) * 3 if len(grid.keys()) > 1 else 1

    stacker = StackingCVRegressor(regressors=regressors,
                                  use_features_in_secondary=True,
                                  shuffle=True,
                                  cv=5,
                                  meta_regressor=ElasticNet(random_state=random_state, alpha=20))

    print('Training model ...')
    t1 = time.time()
    cv = custom_cv(X_train)
    model = BayesSearchCV(stacker, grid, cv=cv, n_jobs=3, random_state=random_state, n_iter=iters)
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
