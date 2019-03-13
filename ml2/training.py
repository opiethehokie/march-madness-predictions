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


from autosklearn.regression import AutoSklearnRegressor
from mlxtend.feature_selection import ColumnSelector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from db.cache import read_model, write_model, model_exists
from ml2.transformers import assign_ranks
from ml2.wrangling import custom_cv


n_jobs = 4

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
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
def manual_regression_model(X, y, random_state=42, tune=True):
    grid = {
        'preprocessing2__n_components': Integer(4, X.shape[1]*2),
        'selection__threshold': Categorical(['median', '.5*median', '.75*median']),
        'regression__alpha': Real(1e-2, 1e+1, prior='log-uniform')
    }

    model = Pipeline(steps=[('engineering', FeatureUnion([('orig', ColumnSelector(2, X.shape[1])),
                                                          ('ranked', FunctionTransformer(assign_ranks, validate=False))
                                                         ], n_jobs=n_jobs)),
                            ('preprocessing1', PowerTransformer(standardize=True)), # minimizes skewness
                            ('preprocessing2', PCA(random_state=random_state)), # helps with multicollinearity problems
                            ('selection', SelectFromModel(LassoCV(cv=3, random_state=random_state, n_jobs=n_jobs))),
                            ('regression', Ridge(random_state=random_state, max_iter=2500))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=10, scoring=make_scorer(r2_score), n_jobs=n_jobs, random_state=random_state, n_iter=25)

    model.fit(X, y)
    return model


def auto_regression_model(X, y, random_state=42, tune=True):
    if model_exists('auto') and not tune:
        return read_model('auto')
    model = AutoSklearnRegressor(time_left_for_this_task=60*60*5, per_run_time_limit=60*10, seed=random_state, resampling_strategy='cv',
                                 resampling_strategy_arguments={'folds': 5})
    model.fit(X.copy(), y.copy())
    model.refit(X.copy(), y.copy())
    print(model.show_models())
    print(model.sprint_statistics())
    write_model(model, 'auto')
    return model


@print_models
def deep_learning_regression_model(X, y, random_state=42, tune=True):
    grid = {
        'regression__hidden_layer_sizes': Categorical([(int(X.shape[0]/(X.shape[1]*10)),),
                                                       (X.shape[1]+2,),
                                                       (int((X.shape[1]+2)*.67),),
                                                       (int((X.shape[1]+2)*.5),)
                                                      ]),
        'regression__activation': Categorical(['logistic', 'relu']),
        'regression__alpha': Real(1e-6, 1e-2, prior='log-uniform')
    }

    model = Pipeline(steps=[('selection', ColumnSelector(cols=[i for i in range(2, X.shape[1]-2)])),
                            ('regression', MLPRegressor(random_state=random_state, max_iter=1000))
                            ])
    if tune:
        model = BayesSearchCV(model, grid, cv=custom_cv(X), scoring=make_scorer(r2_score), n_jobs=n_jobs, random_state=random_state, n_iter=25)

    model.fit(X, y)
    return model
