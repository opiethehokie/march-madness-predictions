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

# from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from ml.regression_stacking_cv_classifier import RegressionStackingCVClassifier
from ml.transformers import ColumnSelector, DebugFeatureProperties
from ml.wrangling import describe_stats, derive_stats, TOURNEY_START_DAY
from ml.util import print_models
# from ratings import off_def, markov


RPI1_COL = 4
RPI2_COL = 5

def rpi_linear_regression(n_jobs=-1):
    return make_pipeline(ColumnSelector(cols=[RPI1_COL, RPI2_COL]),
                         StandardScaler(),
                         LinearRegression(n_jobs=n_jobs))

def rpi_ridge_regression():
    return make_pipeline(ColumnSelector(cols=[RPI1_COL, RPI2_COL]),
                         StandardScaler(),
                         Ridge())

def make_regressor_pipeline(cols, scaler, reg):
    #TODO memory param
    return make_pipeline(ColumnSelector(cols=cols),
                         scaler,
                         reg)

def mov_to_win(label):
    return int(label > 0)

#TODO http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/


@print_models
def train_model(X_train, y_train, random_state, n_jobs=1):

    stacker = RegressionStackingCVClassifier(regressors=[rpi_linear_regression(n_jobs=n_jobs),
                                                         rpi_ridge_regression()
                                                        ],
                                             meta_classifier=LogisticRegression(), #TODO n_jobs=n_jobs
                                             to_class_func=mov_to_win)

    grid = { #'kneighborsclassifier__n_neighbors': [1, 5],
             #'randomforestclassifier__n_estimators': [10, 50],
             #'meta-logisticregression__C': [0.1, 10.0]
           }

    cv = custom_cv(X_train)

    model = GridSearchCV(estimator=stacker, param_grid=grid, cv=cv, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model

def custom_cv(X):
    season_col = X[:, 0]
    seasons = numpy.unique(season_col)
    day_col = X[:, 1]
    return [(numpy.where((season_col != season) | (day_col < TOURNEY_START_DAY))[0],
             numpy.where((season_col == season) & (day_col >= TOURNEY_START_DAY))[0]) for season in seasons[0: -1]]
