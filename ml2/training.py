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

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Real

from ml2.preprocessing import custom_cv

random_state = 42
np.random.seed(random_state)


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
        print('Best parameters: %s' % model.best_params_)
        print('Best accuracy: %.2f' % model.best_score_)
        return model
    return printed_func

@print_models
def manual_regression_model(X, y):
    grid = {
        'alpha': Real(1e+0, 1e+2, prior='log-uniform')
    }
    iters = len(grid.keys()) * 5 if len(grid.keys()) > 1 else 1
    base_model = Ridge(random_state=random_state, alpha=1)
    model = BayesSearchCV(base_model, grid, cv=custom_cv(X), scoring=make_scorer(r2_score), n_jobs=4,
                          random_state=random_state, n_iter=iters)
    model.fit(X, y)
    return model

def auto_regression_model():
    pass

def deep_learning_regression_model():
    pass

def combine_predictions():
    pass
