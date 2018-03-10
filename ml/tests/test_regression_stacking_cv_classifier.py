#   Copyright 2018 Michael Peters
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


import random
import numpy as np
import pandas as pd

from mlxtend.utils import assert_raises
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from ml.regression_stacking_cv_classifier import RegressionStackingCVClassifier

# mostly copied from https://github.com/rasbt/mlxtend/blob/master/mlxtend/classifier/tests/test_stacking_cv_classifier.py

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def to_class(label):
    return random.randint(0, 1)


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    #assert scores_mean == 0.93


def test_StackingClassifier_proba():

    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    #assert scores_mean == 0.93


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    #assert np.allclose(mean_scores, [0.96, 0.96, 0.96, 0.95])


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    #assert scores_mean == 0.93, scores_mean


def test_do_not_stratify():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          stratify=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    #assert scores_mean == 0.94


def test_cross_validation_technique():
    # This is like the `test_do_not_stratify` but instead
    # autogenerating the cross validation strategy it provides
    # a pre-created object
    np.random.seed(123)
    cv = KFold(n_splits=2, shuffle=True)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          cv=cv)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    #assert scores_mean == 0.94


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False)

    assert_raises(NotFittedError,
                  "This RegressionStackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  iris.data)

    assert_raises(NotFittedError,
                  "This RegressionStackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  iris.data)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False,
                                          verbose=3)
    sclf.fit(iris.data, iris.target)


def test_list_of_lists():
    X_list = [i for i in X]
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False,
                                          verbose=0)

    try:
        sclf.fit(X_list, iris.target)
    except TypeError as e:
        assert 'are NumPy arrays. If X and y are lists' in str(e)


def test_pandas():
    X_df = pd.DataFrame(X)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2],
                                          meta_classifier=meta,
                                          to_class_func=to_class,
                                          shuffle=False,
                                          verbose=0)
    try:
        sclf.fit(X_df, iris.target)
    except KeyError as e:
        assert 'are NumPy arrays. If X and y are pandas DataFrames' in str(e)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = RegressionStackingCVClassifier(regressors=[clf1, clf2, clf3],
                                          meta_classifier=lr,
                                          to_class_func=to_class)

    got = sorted(list({s.split('__')[0] for s in sclf.get_params().keys()}))
    expect = ['cv',
              'gaussiannb',
              'kneighborsclassifier',
              'meta-logisticregression',
              'meta_classifier',
              'randomforestclassifier',
              'refit',
              'regressors',
              'shuffle',
              'store_train_meta_features',
              'stratify',
              'to_class_func',
              'verbose']
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = RegressionStackingCVClassifier(regressors=[clf1],
                                          meta_classifier=lr,
                                          to_class_func=to_class)

    params = {'regressors': [[clf1], [clf1, clf2, clf3]]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    assert len(grid.best_params_['regressors']) > 0


def test_train_meta_features_():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    stclf = RegressionStackingCVClassifier(regressors=[knn, gnb],
                                           meta_classifier=lr,
                                           to_class_func=to_class,
                                           store_train_meta_features=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #  test default (class labels)
    stclf = RegressionStackingCVClassifier(regressors=[knn, gnb],
                                           meta_classifier=lr,
                                           to_class_func=to_class,
                                           store_train_meta_features=True)
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)
