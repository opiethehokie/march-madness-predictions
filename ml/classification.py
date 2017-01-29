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


#from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from ml.transformers import (HomeCourtTransformer, ModifiedRPITransformer, OvertimeTransformer, PythagoreanExpectationTransformer,
                             RatingTransformer, SkewnessTransformer, StatTransformer)
from ml.visualizations import plot_auc, plot_confusion_matrix
from ml.wrangling import describe_stats
#from ml.wrangling import derive_stats
#from ratings import markov
from ratings import off_def


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

    random_state = 42 # helps get repeatable results
    n_jobs = -1

    # pipeline guidelines - http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
    # feature selection comparison - http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
    pipe = Pipeline([
        ('preprocessing1', OvertimeTransformer()),
        ('feature_engineering', FeatureUnion([
            ('home_court', HomeCourtTransformer()),
            ('luck', PythagoreanExpectationTransformer()),
            ('sos', ModifiedRPITransformer()),
            #('markov', RatingTransformer(markov.markov_stats, preseason_games)),
            ('offdef', RatingTransformer(off_def.adjust_stats, preseason_games)),
            #('unknown', StatTransformer(derive_stats)),
            ('consistency', StatTransformer(describe_stats)),
        ])),
        ('preprocessing2', Pipeline([
            ('skew', SkewnessTransformer()),
            ('standardize', StandardScaler()),
        ])),
        ('feature_select', RandomizedLogisticRegression(random_state=random_state, n_jobs=1)),
        #('feature_debug', DebugFeatureProperties()),
        ('mlp_classifier', MLPClassifier(random_state=random_state, max_iter=1000))
        #('baseline_classifier', DummyClassifier(strategy='uniform', random_state=random_state))
    ])

    # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # some grid search, some random search, some manual search arrived at these hyper-parameters
    grid = {
        'feature_engineering__luck__exponent': [10.25], # also tried 13.91 and 16.5
        'feature_engineering__sos__weights': [(.15, .15, .7)], # also tried (.25, .25, .5) and (.25, .5, .25)
        'preprocessing2__skew__max_skew': [2.5],
        'preprocessing2__skew__technique': ['log'],
        'mlp_classifier__activation': ['logistic'], # also tried relu
        'mlp_classifier__alpha': [.0001],
        'mlp_classifier__hidden_layer_sizes': [(7)] # also tried 3 and 9
    }

    # 5 or 10 splits is good for balancing bias/variance
    cv = StratifiedKFold(n_splits=5, random_state=random_state)

    #model = GridSearchCV(estimator=pipe, param_grid=grid, scoring='roc_auc', cv=cv, n_jobs=n_jobs)
    #model = RandomizedSearchCV(estimator=pipe, param_distributions=grid, scoring='neg_log_loss', cv=cv,
    #                           n_jobs=n_jobs, random_state=random_state, n_iter=5)
    model = GridSearchCV(estimator=pipe, param_grid=grid, scoring='neg_log_loss', cv=cv, n_jobs=n_jobs)

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_predict_probas = model.predict_proba(X_test)
        print(log_loss(y_test, y_predict_probas))
        plot_auc(y_test, y_predict_probas[:, 1])
        y_predict = model.predict(X_test)
        print(classification_report(y_test, y_predict))
        plot_confusion_matrix(y_test, y_predict)

    return model
