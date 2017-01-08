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


from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from sklearn.decomposition.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals.joblib import Memory
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.transformers import DebugFeatureImportances, DebugFeatureProperties
from ml.transformers import (HomeCourtTransformer, ModifiedRPITransformer, PythagoreanExpectationTransformer, RatingTransformer, 
                             SkewnessTransformer, StatTransformer)
from ml.visualizations import plot_auc, plot_confusion_matrix
from ml.wrangling import derive_stats, describe_stats
from ratings import markov, off_def
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.neural_network.multilayer_perceptron import MLPClassifier


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

    # pipeline guidelines - http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
    # feature selection comparison - http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
    # stacking ensemble works best when first-level classifiers are weakly correlated
    # since our metric is log loss, logistic regression in second-level is good for calibrating the probabilities
    pipe = Pipeline([
        ('feature_engineering', FeatureUnion([
            ('home_court', HomeCourtTransformer()),
            ('pythag', PythagoreanExpectationTransformer()),
            ('sos', ModifiedRPITransformer()),
            ('markov', RatingTransformer(markov.markov_stats, preseason_games)),
            ('offdef', RatingTransformer(off_def.adjust_stats, preseason_games)),
            #('derived_stats', Pipeline([
            #    ('combos', StatTransformer(derive_stats)),
            #    ('prune_garbage', SelectFromModel(ExtraTreesClassifier()))
            #])),
            ('descriptive_stats', Pipeline([
                ('summaries', StatTransformer(describe_stats)),
                ('prune_garbage', SelectFromModel(ExtraTreesClassifier()))
            ]))
        ])),
        ('preprocessing', Pipeline([
            ('skew', SkewnessTransformer()),
            ('standardize', StandardScaler()),
        ])),
        ('feature_select', SelectKBest()), # pca didn't work here
        #('debug', DebugFeatureProperties()),
        #('stacked_classification', StackingClassifier(use_probas=True,
        #                                              average_probas=False,
        #                                              verbose=2,
        #                                              classifiers=[SVC(probability=True),
        #                                                           #KNeighborsClassifier(n_jobs=-1),
        #                                                           #RandomForestClassifier(n_jobs=-1),
        #                                                           #GradientBoostingClassifier(),
        #                                                           #GaussianNB(),
        #                                                           #DNNClassifier(feature_columns=[infer_real_valued_columns_from_input()], n_classes=2),
        #                                                           ],
        #                                              meta_classifier=LogisticRegression(n_jobs=-1))) #TODO switch this to StackingCVClassifier #TODO should be roughly .48 - .57, 2001 had a lot of upsets
        #])
        #('svm', SVC(probability=True)) # .489
        #('bayes', LinearDiscriminantAnalysis()) # .504
        #('mlp', MLPClassifier()), # .451
        #('knn', KNeighborsClassifier()), # .563
        ('rf', RandomForestClassifier())
    #], memory=Memory(cachedir='.cache', verbose=0))
    ])

    # variance = underfitting
    # bias = overfitting
    # regularization decreases variance and increases bias in linear models and neural networks
    # bagging combines many high variance models to create better fit
    # boosting combines many high bias models to create better fit
    # bias increases and variance decreases with k in knn
    grid = {
        'feature_engineering__pythag__exponent': [10.25], #also tried 13.91 and 16.5
        'feature_engineering__sos__weights': [(.15, .15, .7)], # also tried (.25, .25, .5) and (.25, .5, .25)
        #'feature_engineering__derived_stats__prune_garbage__threshold': [.007], # no features here was best
        'feature_engineering__descriptive_stats__prune_garbage__threshold': [.007], # also tried .006 and .008
        'feature_engineering__markov__last_x_games': [0], # also tried 7 and 14
        'feature_engineering__offdef__last_x_games': [0], # also tried 5 and 10
        'preprocessing__skew__max_skew': [.75], # also tried 3
        'preprocessing__skew__technique': ['log'], # also tried 'sqrt', 'boxcox', and None
        'feature_select__k': [40],
        #'svm__kernel': ['linear'], # also tried rbf
        #'mlp__hidden_layer_sizes': [(15,)], # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        #'knn__n_neighbors': [50],
        'rf__n_estimators': [300], # also tried 100 and 500
        
        'rf__max_depth': [5, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 5],
        
        #'gb__max_depth': [3, 5, 7],
        #'gb__subsample': [.6, .8, 1],
        #'gb__loss': ['deviance', 'exponential'],
        #'gb__min_samples_split': [2, 5, 10],
        #'gb__max_features': ['log2', 'sqrt', None],
        #'gb__min_samples_leaf': [1, 2, 5],
        #'gb__n_estimators': [100, 300, 500],
        
        
        #'stacked_classification__meta-logisticregression__C': [.01, .1, 1]
        #'stacked_classification__meta-logisticregression__penalty': ['l1', 'l2']
    }

    cv = StratifiedKFold(n_splits=2)
    #cv = TimeSeriesSplit(n_splits=2) #TODO 5 or 10 splits is good for balancing bias/variance
    #model = GridSearchCV(estimator=pipe, param_grid=grid, cv=cv, scoring='neg_log_loss', n_jobs=-1) #TODO different metric
    model = GridSearchCV(estimator=pipe, param_grid=grid, cv=cv, n_jobs=-1) #TODO different metric

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_predict_probas = model.predict_proba(X_test)
        print(log_loss(y_test, y_predict_probas))
        plot_auc(y_test, y_predict_probas[:, 1])
        y_predict = model.predict(X_test)
        plot_confusion_matrix(y_test, y_predict)

    return model
