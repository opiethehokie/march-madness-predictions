#   Copyright 2016 Michael Peters
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


from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals.joblib import Memory
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from ml.pipeline import CachedPipeline, FeatureUnion, Pipeline
from ml.transformers import DebugShape, DebugFeatureImportances
from ml.transformers import (HomeCourtTransformer, StatTransformer, ModifiedRPITransformer,
                             MovStdDevTransformer, PythagoreanExpectationTransformer, RatingTransformer)
from ml.wrangling import derive_stats, describe_stats
from ratings import markov, off_def


# annotate function with @print_models to see CV details
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
def train_stacked_model(preseason_games, X_train, X_test, y_train, y_test):

    pipe = CachedPipeline([
        ('feature_engineering', FeatureUnion([
            ('home_court', HomeCourtTransformer()),
            ('results', Pipeline([
                ('types', FeatureUnion([
                    ('mov', MovStdDevTransformer()),
                    ('pythag', PythagoreanExpectationTransformer()),
                    ('sos', ModifiedRPITransformer()) #TODO stat graphs here
                ])),
                ('scaling', StandardScaler())
            ])),
            # feature selection comparison - http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
            ('derived_stats', Pipeline([
                ('combos', StatTransformer(derive_stats)),
                ('selection', SelectFromModel(ExtraTreesClassifier())),
                ('scaling', Normalizer())
            ])),
            ('descriptive_stats', Pipeline([
                ('summaries', StatTransformer(describe_stats)),
                ('scaling', StandardScaler()),
                ('selection', RFE(LinearSVC(dual=False)))
            ])),
            ('adj_stats1', Pipeline([
                ('markov', RatingTransformer(markov.markov_stats, preseason_games)),
                ('scaling', Normalizer(norm='l1')),
                ('selection', RandomizedLogisticRegression()) # stability selection
            ])),
            ('adj_stats2', Pipeline([
                ('offdef', RatingTransformer(off_def.adjust_stats, preseason_games)),
                ('scaling', Normalizer(norm='l1')),
                ('selection', RandomizedLogisticRegression()),
                ('debug', DebugShape())
            ]))
        ])),
        #TODO scatter matrix here
        # stacking ensemble works best when first-level classifiers are weakly correlated
        # since our metric is log loss, logistic regression in second-level is good for calibrating the probabilities
        ('stacked_classification', StackingClassifier(use_probas=True,
                                                      average_probas=False,
                                                      verbose=2,
                                                      classifiers=[SVC(probability=True, kernel='linear'),
                                                                   KNeighborsClassifier(n_jobs=-1),
                                                                   RandomForestClassifier(n_jobs=-1),
                                                                   GradientBoostingClassifier(),
                                                                   GaussianNB()
                                                                   ],
                                                      meta_classifier=LogisticRegression(n_jobs=-1))) #TODO switch this to StackingCVClassifier
        ], memory=Memory(cachedir='.cache', verbose=0))

    # variance = underfitting
    # bias = overfitting
    # regularization decreases variance and increases bias in linear models and neural networks
    # bagging combines many high variance models to create better fit
    # boosting combines many high bias models to create better fit
    # bias increases and variance decreases with k in knn
    grid = {
        'feature_engineering__results__types__mov__max_mov': [20, None],
        'feature_engineering__results__types__pythag__exponent': [10.25, 13.91, 16.5],
        'feature_engineering__results__types__sos__weights': [(.15, .15, .7), (.25, .5, .25), (.25, .25, .5)],
        'feature_engineering__adj_stats1__markov__last_x_games': [5, 10, 20],
        'feature_engineering__adj_stats2__offdef__last_x_games': [5, 10, 20],
        'feature_engineering__derived_stats__selection__threshold': [.006, .008],
        'feature_engineering__descriptive_stats__selection__n_features_to_select': [5, 10],
        'feature_engineering__adj_stats1__selection__selection_threshold': [.25, .5],
        'feature_engineering__adj_stats2__selection__selection_threshold': [.25, .5],
        'stacked_classification__svc__C': [.1, 1, 10],
        'stacked_classification__kneighborsclassifier__n_neighbors': [2, 4, 8, 16],
        'stacked_classification__kneighborsclassifier__p': [1, 2, 3],
        'stacked_classification__randomforestclassifier__n_estimators': [120, 300, 500],
        'stacked_classification__randomforestclassifier__max_depth': [5, 8, 15, 25, None],
        'stacked_classification__randomforestclassifier__min_samples_split': [2, 5, 10, 15],
        'stacked_classification__randomforestclassifier__min_samples_leaf': [1, 2, 5, 10],
        'stacked_classification__randomforestclassifier__max_features': ['log2', 'sqrt', None],
        'stacked_classification__gradientboostingclassifier__max_depth': [3, 5, 7, 9, 12, 18],
        'stacked_classification__gradientboostingclassifier__subsample': [.7, .9, 1],
        'stacked_classification__gradientboostingclassifier__loss': ['deviance', 'exponential'],
        'stacked_classification__gradientboostingclassifier__min_samples_split': [2, 5, 10, 15],
        'stacked_classification__gradientboostingclassifier__max_features': ['log2', 'sqrt', None],
        'stacked_classification__gradientboostingclassifier__min_samples_leaf': [1, 2, 5, 10],
        'stacked_classification__gradientboostingclassifier__n_estimators': [120, 300, 500],
        'stacked_classification__meta-logisticregression__C': [0.1, 1, 10.0]
    }

    cv = TimeSeriesSplit(n_splits=5) # 5 or 10 splits is good for balancing bias/variance
    model = GridSearchCV(estimator=pipe, param_grid=grid, cv=cv, scoring='neg_log_loss', n_jobs=-1)

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_predict = model.predict_proba(X_test)
        print(log_loss(y_test, y_predict))
        #TODO roc curve here
        #TODO confusion matrix here

    return model
