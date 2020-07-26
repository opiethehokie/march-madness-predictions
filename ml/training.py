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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, KBinsDiscretizer
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
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
        'engineering__bin__n_bins': Integer(5, 10),
        'classification__C': Real(1e-3, 1e-1, prior='log-uniform'),
    }

    model = Pipeline(steps=[('preprocessing', PowerTransformer()),
                            ('engineering', FeatureUnion([('pca', PCA(random_state=rs, n_components=None)),
                                                          ('cluster', FeatureAgglomeration(n_clusters=10)),
                                                          ('bin', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')),
                                                         ])),
                            ('selection', SelectKBest(k=75)),
                            ('classification', LogisticRegression(C=.1, random_state=rs, solver='saga', penalty='l2', max_iter=10000))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=64)
    if fit:
        model.fit(X, y)
    return model


@print_models
def boosting_model(X, y, cv=10, rs=42, tune=True, fit=True):
    grid = {
        'selection__k': Integer(75, 125),
        'classification__max_depth': Integer(3, 5),
        'classification__min_child_weight': Integer(1, 3),
        'classification__n_estimators': Integer(500, 1000),
        'classification__learning_rate': Real(1e-2, 1e-1, prior='log-uniform'), # eta
        'classification__reg_alpha': Real(1e-2, 1, prior='log-uniform'), # l1
        'classification__reg_lambda': Real(1e-2, 1, prior='log-uniform'), # l2
        'classification__booster': Categorical(['gbtree', 'dart', 'gblinear']),
        'classification__importance_type': Categorical(['gain', 'weight', 'cover', 'total_gain', 'total_cover'])
    }

    model = Pipeline(steps=[('selection', SelectKBest(k=100)),
                            ('classification', XGBClassifier(objective='binary:logistic', random_state=rs, max_depth=5, learning_rate=.05,
                                                             min_child_weight=1, n_estimators=500, gamma=0, subsample=1, colsample_bytree=.6,
                                                             reg_alpha=1, reg_lambda=1))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=128)
    if fit:
        model.fit(X, y)
    return model


@print_models
def neural_network_model(X, y, cv=5, rs=42, tune=True, fit=True):
    grid = {
        'selection__threshold': Categorical(['mean', 'median']),
        'classification__batch_size': Integer(64, 256, prior='log-uniform', base=2),
        'classification__drop': Real(.1, .5),
        'classification__opt': Categorical(['adam', 'adagrad']),
        'classification__act': Categorical(['tanh', 'relu']),
        'classification__reg': Real(1e-5, 1e-4, prior='log-uniform'),
        'classification__hls': Integer(int(X.shape[1]/4), X.shape[1]*2)
    }

    def create_mlp(init='normal', drop=.1, opt='adam', act='relu', reg=1e-4, hls=64):
        mlp = Sequential()
        mlp.add(Dense(hls, activation=act, kernel_initializer=init, activity_regularizer=l2(reg)))
        mlp.add(Dropout(drop, seed=rs))
        mlp.add(Dense(1, activation='sigmoid'))
        mlp.compile(loss='binary_crossentropy', optimizer=opt, metrics=[])
        return mlp

    callback = EarlyStopping(monitor='loss', patience=10)

    model = Pipeline(steps=[('preprocessing', MinMaxScaler()),
                            ('classification', KerasClassifier(build_fn=create_mlp, epochs=500, batch_size=256, verbose=0))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=1, random_state=rs, n_iter=32,
                              fit_params=dict(classification__callbacks=[callback]))
        if fit:
            model.fit(X, y)
    else:
        if fit:
            model.fit(X, y, classification__callbacks=[callback])
    return model


@print_models
def bayesian_model(X, y, cv=5, rs=42, tune=True, fit=True):
    grid = {
        'selection__k': Integer(25, X.shape[1]),
        'classification__solver': Categorical(['lsqr', 'eigen']),
        'classification__shrinkage': Categorical([None, 'auto'])
    }

    model = Pipeline(steps=[('selection', SelectKBest(k=50)),
                            ('classification', LinearDiscriminantAnalysis(solver='svd'))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=32)
    if fit:
        model.fit(X, y)
    return model


@print_models
def stacked_model(X, y, cv=2, rs=42):
    clfs = [linear_model(X, y, rs=rs, tune=False, fit=False),
            neural_network_model(X, y, rs=rs, tune=False, fit=False),
            boosting_model(X, y, rs=rs, tune=False, fit=False),
            bayesian_model(X, y, rs=rs, tune=False, fit=False)
           ]
    mclf = LogisticRegression(penalty='l2', random_state=rs, C=.01, solver='saga')
    model = StackingCVClassifier(classifiers=clfs, use_probas=True, use_features_in_secondary=False, meta_classifier=mclf,
                                 random_state=rs, cv=cv, n_jobs=1)
    model.fit(X, y)
    return model
