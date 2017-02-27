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


from sklearn.cluster import FeatureAgglomeration
#from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSlit
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from ml.transformers import (HomeCourtTransformer, ModifiedRPITransformer, OvertimeTransformer, PythagoreanExpectationTransformer,
                             RatingTransformer, SimpleRatingTransformer, SkewnessTransformer, StatTransformer)
#from ml.transformers import DebugFeatureProperties
from ml.visualizations import plot_auc, plot_confusion_matrix
from ml.wrangling import describe_stats, derive_stats
from ratings import off_def, markov


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
def train_model(preseason_games, X_train, X_test, y_train, y_test, random_state):

    n_jobs = -1

    # pipeline guidelines - http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
    # feature selection comparison - http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
    pipe = Pipeline([
        ('preprocess_overtime', OvertimeTransformer()),
        ('preprocess_hca', HomeCourtTransformer()),
        ('feature_engineering', FeatureUnion([
            ('luck', PythagoreanExpectationTransformer()),
            ('sos', ModifiedRPITransformer()),
            ('consistency', StatTransformer(describe_stats)),
            ('offdef', RatingTransformer(off_def.adjust_stats, preseason_games)),
            ('markov', SimpleRatingTransformer(markov.markov_stats, preseason_games)),
            ('unknown', Pipeline([
                ('pairwise_combos', StatTransformer(derive_stats)),
                ('dimension_reduction', FeatureAgglomeration())
            ])),
        ])),
        ('preprocess_skew', SkewnessTransformer()),
        ('standardize', StandardScaler(with_std=False)),
        ('feature_selection', RandomizedLogisticRegression(random_state=random_state, n_jobs=1)),
        #('feature_debug', DebugFeatureProperties()),
        #('baseline_classifier', DummyClassifier(strategy='uniform', random_state=random_state))
        ('mlp_classifier', MLPClassifier(random_state=random_state, max_iter=1000))
    ])

    # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # some grid search, some random search, some manual search arrived at these hyper-parameters
    grid = {
        'preprocess_hca__factor': [.96],
        'feature_engineering__luck__exponent': [13.91], # also tried 10.25 and 16.5
        'feature_engineering__sos__weights': [(.15, .15, .7)], # also tried (.25, .25, .5) and (.25, .5, .25)
        'feature_engineering__unknown__dimension_reduction__n_clusters': [50],
        'feature_engineering__unknown__dimension_reduction__affinity': ['cosine'],
        'feature_engineering__unknown__dimension_reduction__linkage': ['average'],
        'preprocess_skew__max_skew': [2.5],
        'mlp_classifier__activation': ['logistic'],
        'mlp_classifier__alpha': [.0001],
        'mlp_classifier__hidden_layer_sizes': [(7)] # n+1 / 2 or 2n/3 + 1 or sqrt(n+1) or samples / 10 * n+1
    }

    # 5 or 10 splits is good for balancing bias/variance
    cv = StratifiedKFold(n_splits=5, random_state=random_state)
    #cv = TimeSeriesSplit(n_splits=5)

    #model = RandomizedSearchCV(estimator=pipe, param_distributions=grid, scoring='neg_log_loss', cv=cv,
    #                           n_jobs=n_jobs, random_state=random_state, n_iter=25)
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
