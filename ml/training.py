#   Copyright 2016-2020 Michael Peters
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


from mlxtend.classifier import StackingCVClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, KBinsDiscretizer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier


n_jobs = 4

scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)


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


@print_models
def linear_model(X, y, cv=10, rs=42, tune=True, fit=True):
    grid = {
        'engineering__pca__n_components': Integer(2, int(X.shape[1])),
        'engineering__cluster__n_clusters': Integer(2, 20),
        #'engineering__bin__encode': Categorical(['ordinal', 'onehot']),
        'engineering__bin__n_bins': Integer(5, 10),
        #'selection1__threshold': Real(0, .2),
        'selection2__k': Integer(50, 100),
        'classification__C': Real(1e-3, 1e-1, prior='log-uniform'),
        #'classification__penalty': Categorical(['l1', 'l2', 'elasticnet'])
    }

    model = Pipeline(steps=[('preprocessing', PowerTransformer()),
                            ('engineering', FeatureUnion([('pca', PCA(random_state=rs, n_components=None)),
                                                          ('cluster', FeatureAgglomeration(n_clusters=10)),
                                                          ('bin', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')),
                                                         ])),
                            ('selection1', VarianceThreshold(threshold=0)),
                            ('selection2', SelectKBest(k=75)),
                            ('classification', LogisticRegression(C=.01, random_state=rs, solver='saga', penalty='l2', max_iter=10000))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=64)
    if fit:
        model.fit(X, y)
    return model


@print_models
def tree_model(X, y, cv=10, rs=42, tune=True, fit=True):
    grid = {
        'selection__k': Integer(75, 125),
        'classification__max_depth': Integer(3, 5),
        'classification__min_child_weight': Integer(1, 3),
        'classification__n_estimators': Integer(500, 1000),
        'classification__learning_rate': Real(1e-2, 1e-1, prior='log-uniform'),
        'classification__gamma': Real(0, .2),
        'classification__subsample': Real(.8, 1),
        'classification__colsample_bytree': Real(.6, .8),
        'classification__reg_alpha': Real(1e-2, 1e1, prior='log-uniform')
    }

    model = Pipeline(steps=[('selection', SelectKBest(k=100)),
                            ('classification', XGBClassifier(objective='binary:logistic', random_state=rs, max_depth=5, learning_rate=.05,
                                                             min_child_weight=1, n_estimators=500, gamma=0, subsample=1, colsample_bytree=.6,
                                                             reg_alpha=.1))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=128)
    if fit:
        model.fit(X, y)
    return model


@print_models
def neural_network_model(X, y, cv=5, rs=42, tune=True, fit=True):
    grid = {
        #'classification__activation': Categorical(['relu', 'tanh', 'logistic']),
        'classification__alpha': Real(1e-6, 1e-2, prior='log-uniform'),
        'classification__learning_rate_init': Real(1e-5, 1e-1, prior='log-uniform'),
        'classification__batch_szie': Integer(200, 400)
    }

    #hls = (int(X.shape[0]/(X.shape[1]*10)),)
    #hls = (X.shape[1]+2,)
    #hls = (int((X.shape[1]+2)*.67),)
    hls = (int((X.shape[1]+2)*.5),)
    #hls = (X.shape[1]+2, X.shape[1]+2)
    #hls = (X.shape[1]+2, int((X.shape[1]+2)*.5))

    model = Pipeline(steps=[('preprocessing', MinMaxScaler()),
                            ('classification', MLPClassifier(activation='relu', alpha=.0001, early_stopping=True, learning_rate_init=.001,
                                                             hidden_layer_sizes=hls, random_state=rs, max_iter=10000, batch_size=300))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=64)
    if fit:
        model.fit(X, y)
    return model


@print_models
def stacked_model(X, y, cv=2, rs=42):
    clfs = [linear_model(X, y, rs=rs, tune=False, fit=False),
            neural_network_model(X, y, rs=rs, tune=False, fit=False),
            tree_model(X, y, rs=rs, tune=False, fit=False)
           ]
    mclf = LogisticRegression(penalty='l2', random_state=rs, C=.1, solver='saga')
    model = StackingCVClassifier(classifiers=clfs, use_probas=True, use_features_in_secondary=False, meta_classifier=mclf,
                                 random_state=rs, cv=cv, n_jobs=n_jobs)
    model.fit(X, y)
    return model
