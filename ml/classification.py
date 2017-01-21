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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.stacker import StackingClassifier
from ml.transformers import DebugFeatureProperties
from ml.transformers import (HomeCourtTransformer, ModifiedRPITransformer, OvertimeTransformer, PythagoreanExpectationTransformer,
                             RatingTransformer, SkewnessTransformer, StatTransformer)
from ml.visualizations import plot_auc, plot_confusion_matrix
from ml.wrangling import derive_stats, describe_stats
from ratings import markov, off_def


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

    seed = 1 # helps get repeatable results

    # pipeline guidelines - http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
    # feature selection comparison - http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
    # stacking ensemble works best when first-level classifiers are weakly correlated but accuracy is similar
    # since our metric is log loss, logistic regression in second-level is good for calibrating the probabilities
    pipe = Pipeline([
        ('preprocessing1', OvertimeTransformer()),
        ('feature_engineering', FeatureUnion([
            ('home_court', HomeCourtTransformer()),
            ('pythag', PythagoreanExpectationTransformer()),
            ('sos', ModifiedRPITransformer()),
            ('markov', RatingTransformer(markov.markov_stats, preseason_games)),
            ('offdef', RatingTransformer(off_def.adjust_stats, preseason_games)),
            ('derived_stats', Pipeline([
                ('combos', StatTransformer(derive_stats)),
                ('prune_garbage', SelectFromModel(ExtraTreesClassifier()))
            ])),
            ('descriptive_stats', Pipeline([
                ('summaries', StatTransformer(describe_stats)),
                ('prune_garbage', SelectFromModel(ExtraTreesClassifier(n_jobs=-1)))
            ]))
        ])),
        ('preprocessing2', Pipeline([
            ('skew', SkewnessTransformer()),
            ('standardize', StandardScaler()),
        ])),
        ('feature_select', SelectKBest()),
        ('describe_features', DebugFeatureProperties()),
        ('stacked_classification', StackingClassifier(use_probas=True,
                                                      use_features_in_secondary=True,
                                                      average_probas=False,
                                                      classifiers=[SVC(random_state=seed, probability=True),
                                                                   LinearDiscriminantAnalysis(),
                                                                   KNeighborsClassifier(n_jobs=-1),
                                                                   MLPClassifier(random_state=seed, max_iter=500),
                                                                   GaussianNB()],
                                                      meta_classifier=LogisticRegression(random_state=seed, n_jobs=-1)))
    ])

    
    # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    grid = {
        'feature_engineering__pythag__exponent': [10.25], # also tried 13.91 and 16.5
        'feature_engineering__sos__weights': [(.15, .15, .7)], # also tried (.25, .25, .5) and (.25, .5, .25)
        'feature_engineering__derived_stats__prune_garbage__threshold': [.007], # based on my DebugFeatureImportances transformer
        'feature_engineering__descriptive_stats__prune_garbage__threshold': [.007], # based on my DebugFeatureImportances transformer
        'feature_engineering__markov__last_x_games': [0],
        'feature_engineering__offdef__last_x_games': [0],
        'preprocessing2__skew__max_skew': [.5, .75],
        'preprocessing2__skew__technique': ['log'],
        'feature_select__k': [35, 40, 45],
        'stacked_classification__svc__kernel': ['linear'],
        'stacked_classification__kneighborsclassifier__n_neighbors': [25], # smaller k gave really bad results
        'stacked_classification__mlpclassifier__activation': ['logistic'],
        'stacked_classification__mlpclassifier__hidden_layer_sizes': [(9), (21), (29)], # also tried 21 and 29
        'stacked_classification__meta-logisticregression__C': [.001],
        'stacked_classification__meta-logisticregression__penalty': ['l2']
    }

    # 5 or 10 splits is good for balancing bias/variance
    cv = StratifiedKFold(n_splits=5, random_state=seed)
    #cv = TimeSeriesSplit(n_splits=3)

    model = GridSearchCV(estimator=pipe, param_grid=grid, scoring='roc_auc', cv=cv, n_jobs=-1)
    #model = GridSearchCV(estimator=pipe, param_grid=grid, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    #model = RandomizedSearchCV(estimator=pipe, param_distributions=grid, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=seed, n_iter=5)

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_predict_probas = model.predict_proba(X_test)
        print(log_loss(y_test, y_predict_probas))
        plot_auc(y_test, y_predict_probas[:, 1])
        y_predict = model.predict(X_test)
        plot_confusion_matrix(y_test, y_predict)

    return model
