#   Copyright 2016-2019 Michael Peters
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
import pickle

from autosklearn.regression import AutoSklearnRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

from db.cache import read_model, write_model, model_exists
from ml2.postprocessing import mov_to_win_percent


#TODO include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
#pylint: disable=function-redefined
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, _):
        raise BaseException('Use newer skopt')


def print_models(func):
    def printed_func(*args, **kwargs):
        model = func(*args, **kwargs)
        cv_keys = ('mean_test_score', 'std_test_score', 'params')
        for r, _ in enumerate(model.cv_results_['mean_test_score']):
            print("%0.3f +/- %0.2f %r" % (model.cv_results_[cv_keys[0]][r],
                                          model.cv_results_[cv_keys[1]][r] / 2.0,
                                          model.cv_results_[cv_keys[2]][r]))
        print('Training metric: %s' % model.scorer_)
        print('Best training parameters: %s' % model.best_params_)
        print('Best training accuracy: %.2f' % model.best_score_)
        return model
    return printed_func

@print_models
def manual_regression_model(X, y, random_state):
    grid = {
        'regression__alpha': Real(1e+0, 1e+2, prior='log-uniform')
    }
    iters = len(grid.keys()) * 5 if len(grid.keys()) > 1 else 1
    model = Pipeline(steps=[#TODO feature engineering
                            #TODO feture selection
                            #TODO preprocessing
                            ('regression', Ridge(random_state=random_state))
                           ])
    model = BayesSearchCV(model, grid, cv=10, scoring=make_scorer(r2_score), n_jobs=4, random_state=random_state, n_iter=iters)
    model.fit(X, y)
    return model

def auto_regression_model(X, y):
    if model_exists('auto'):
        return read_model('auto')
    model = AutoSklearnRegressor(resampling_strategy='cv', resampling_strategy_arguments={'folds': 5})
    model.fit(X.copy(), y.copy())
    model.refit(X.copy(), y.copy())
    print(model.show_models())
    print(model.sprint_statistics())
    write_model(model, 'auto')
    return model

@print_models
def deep_learning_regression_model(X, y, random_state):
    grid = {
        'regression__hidden_layer_sizes': Categorical([(100, 50)]), #TODO
        'regression__activation': Categorical(['relu']),
        'regression__alpha': Real(1e-7, 1e-5, prior='log-uniform'),
    }
    iters = len(grid.keys()) * 5 if len(grid.keys()) > 1 else 1
    model = Pipeline(steps=[('normalization', Normalizer()),
                            ('regression', MLPRegressor(random_state=random_state))
                           ])
    model = BayesSearchCV(model, grid, cv=10, scoring=make_scorer(r2_score), n_jobs=4, random_state=random_state, n_iter=iters)
    model.fit(X, y)
    return model

def average_predictions(models, X, mov_std):
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    results = np.mean(np.array(predictions), axis=0)
    return [mov_to_win_percent(yi, mov_std) for yi in results]
