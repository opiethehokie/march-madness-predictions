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


#from autosklearn.classification import AutoSklearnClassifier
from eli5.sklearn.permutation_importance import PermutationImportance
from eli5 import transform_feature_names
from feature_engine.discretisers import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

#import autosklearn.metrics

#from db.cache import read_model, write_model, model_exists


n_jobs = 4

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
#pylint: disable=function-redefined
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, _):
        raise BaseException('Use newer skopt')

def print_models(func):
    def printed_func(*args, **kwargs):
        model = func(*args, **kwargs)
        if hasattr(model, 'cv_results_'):
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

@transform_feature_names.register(ColumnSelector)
def feature_names(transformer, in_names=None):
    return ['orig' + str(i) for i in range(0, len(transformer.cols))]

@transform_feature_names.register(PCA)
def feature_names(transformer, in_names=None):
    return ['pca' + str(i) for i in range(0, transformer.n_components)]

@transform_feature_names.register(FeatureAgglomeration)
def feature_names(transformer, in_names=None):
    return ['cluster' + str(i) for i in range(0, transformer.n_clusters)]

@print_models
def linear_model(X, y, cv=10, rs=42, tune=True):
    grid = {
        'engineering__pca__n_components': Integer(2, 8),
        'engineering__cluster__n_clusters': Integer(2, 8),
        'selection__estimator__estimator__C': Real(1e-2, 1, prior='log-uniform'),
        'selection__threshold': Real(1e-4, 1e-2, prior='log-uniform'),
        'classification__C': Real(1e-3, 1e-1, prior='log-uniform')
    }

    sparse_features = LinearSVC(C=.1, random_state=rs, penalty='l1', dual=False, max_iter=10000)
    feature_importances = PermutationImportance(sparse_features, cv=None, random_state=rs) #TODO comment how does this relate to SHAP?

    model = Pipeline(steps=[('preprocessing', StandardScaler()),
                            #('preprocessing', PowerTransformer()), #TODO
                            ('engineering', FeatureUnion([('cols', ColumnSelector(cols=[i for i in range(0, X.shape[1])])),
                                                          # dimension reduction helps with multicollinearity
                                                          ('pca', PCA(random_state=rs, n_components=2)),
                                                          ('cluster', FeatureAgglomeration(n_clusters=2)),
                                                          #TODO
                                                          #('bins', EqualWidthDiscretiser(bins=10)),
                                                          ('bins', EqualFrequencyDiscretiser(q=10))
                                                         ])),
                            #TODO correlation based feature selection ANOVA/Kendall
                            ('selection', SelectFromModel(feature_importances, threshold=.0005)), # mean decrease accuracy (MDA)
                            # logistic regression shuffles which helps with autocorrelation
                            ('classification', LogisticRegression(C=.001, random_state=rs, dual=False, solver='saga', penalty='l2', max_iter=10000)) #TODO l1 would do feature selection
                           ])

    if tune:
        scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=64)

    model.fit(X, y)
    return model

@print_models
def mov_regression_model(X, y):
    #TODO https://github.com/opiethehokie/march-madness-predictions/commit/569aedb0ae1dc44ec462add654671f49e0e7236a
    #TODO mov should be capped
    #TODO correlation based feature selection pearson/spearman
    pass

@print_models
def tree_model(X, y):
    #TODO xgboost
    pass

# Linux only
#def automl_model(X, y, rs=42, tune=True):
#    name = 'auto'
#    if model_exists(name) and not tune:
#        return read_model(name)
#    model = AutoSklearnClassifier(time_left_for_this_task=60*60*8, per_run_time_limit=60*15, seed=rs, resampling_strategy='cv',
#                                  resampling_strategy_arguments={'folds': 5})
#    model.fit(X.copy(), y.copy(), metric=autosklearn.metrics.log_loss)
#    model.refit(X.copy(), y.copy())
#    print(model.show_models())
#    print(model.sprint_statistics())
#    write_model(model, name)
#    return model

@print_models
def neural_network_model(X, y, rs=42, tune=True):
    grid = {
        'classification__activation': Categorical(['relu', 'tanh', 'logistic']),
        'classification__hidden_layer_sizes': Categorical([(int(X.shape[0]/(X.shape[1]*10)),),
                                                           (X.shape[1]+2,),
                                                           (int((X.shape[1]+2)*.67),),
                                                           (int((X.shape[1]+2)*.5),),
                                                           (X.shape[1]+2, X.shape[1]+2),
                                                           (X.shape[1]+2, int((X.shape[1]+2)*.5))
                                                          ]),
        'classification__alpha': Real(1e-6, 1e-2, prior='log-uniform'),
        'classification__learning_rate_init': Real(1e-6, 1e-2, prior='log-uniform')
    }

    model = Pipeline(steps=[('preprocessing', MinMaxScaler()),
                            ('classification', MLPClassifier(activation='tanh', alpha=.01, early_stopping=True, learning_rate_init=.01,
                                                             hidden_layer_sizes=(34,), random_state=rs, max_iter=10000))
                           ])

    if tune:
        scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        model = BayesSearchCV(model, grid, cv=5, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=64)

    model.fit(X, y)
    return model

@print_models
def genetic_model(X, y):
    #TODO https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-classifier
    pass

@print_models
def stacked_model(X, y, rs=42):
    clfs = [linear_model(X, y, rs=rs, tune=False),
            neural_network_model(X, y, rs=rs, tune=False),
           ]
    #TODO try EnsembleVoteClassifier too, this could just be ensemble model
    model = StackingCVClassifier(classifiers=clfs, use_features_in_secondary=False, use_probas=True,
                                 meta_classifier=CalibratedClassifierCV(GaussianNB(), cv=5)) #TODO could this be bagged?
    model.fit(X, y)
    return model

@print_models
def bagged_model(X, y):
    #TODO https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    pass

@print_models
def genetic_model(X, y):
    #TODO https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-classifier
    pass
