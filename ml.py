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

import csv
import pandas
import datetime
import numpy

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.feature_selection.univariate_selection import f_classif
from sklearn import neural_network
from sklearn.pipeline import FeatureUnion
from sklearn import cluster

from statistics import logloss
from util import possible_tourney_matchups
from constants import TOURNEY_START_DAY


def write_predictions(pmatchups, confidences):
    with open('submissions/submission.csv', 'wb') as csvfile:
        fieldnames = ['id', 'pred']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(pmatchups)):            
            writer.writerow({ 'id': pmatchups[i], 'pred': confidences[i] })
            
def score_predictions(pmatchups, predictions):
    actual = []
    testable_predictions = []
    with open('data/tourney_detailed_results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['Season'])
            wteam = int(row['Wteam'])
            lteam = int(row['Lteam'])
            daynum = int(row['Daynum'])
            if daynum >= TOURNEY_START_DAY:
                if wteam < lteam:
                    matchup = str(season) + '_' + str(wteam) + '_' + str(lteam)
                    result = 1
                else:
                    matchup = str(season) + '_' + str(lteam) + '_' + str(wteam)
                    result = 0
                if matchup in pmatchups:
                    i = pmatchups.index(matchup)
                    actual.append(result)
                    testable_predictions.append(predictions[i])
    return logloss(actual, testable_predictions)

def data(pseasons):
    known_df = pandas.read_csv('data/train.csv', header=None)
    #print known_df.shape
    filtered_known_df = known_df[(~known_df.iloc[:,0].isin(pseasons)) | (known_df.iloc[:,2] < TOURNEY_START_DAY)]
    #print filtered_known_df.shape
    features = filtered_known_df.iloc[:,3:]
    #print features.shape
    labels = filtered_known_df.iloc[:,2]
    
    unknown_df = pandas.read_csv('data/predict.csv', header=None)
    #print unknown_df.shape
    filtered_unknown_df = unknown_df[unknown_df.iloc[:,0].isin(pseasons)]
    #print filtered_unknown_df.shape
    predict_features = filtered_unknown_df.iloc[:,3:]
    #print predict_features.shape
    return features.values, labels.values, predict_features.values


predict_seasons = [#2012, 
                   #2013, 
                   #2014, 
                   #2015,
                   2016
                   ]
X, y, predict_X = data(predict_seasons)

print(datetime.datetime.now())

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
predict_X = scaler.transform(predict_X)

# rpi, sos, pythag, consistency, oefgp, sr, defgp, dftr, oftr, top, ar, adj-wins, br, orbp, oeff, pir, drbp, deff
#kbest = feature_selection.SelectKBest(f_classif, k='all').fit(X, y)
#print kbest.scores_

selecter = FeatureUnion([('select_poly', preprocessing.PolynomialFeatures(degree=2)),
                         ('select_kbest', feature_selection.SelectKBest(f_classif, k='all')), 
                         ('select_pca', decomposition.PCA(n_components=2)),
                         ('select_cluster', cluster.FeatureAgglomeration(n_clusters=2))
                         ]).fit(X, y)
X = selecter.transform(X)
predict_X = selecter.transform(predict_X)

kbest = feature_selection.SelectKBest(f_classif, k=40).fit(X, y)
X = kbest.transform(X)
predict_X = kbest.transform(predict_X)

# blending - http://mlwave.com/kaggle-ensembling-guide/
# code copied from https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
# for majority vote or average use VotingClassifier

n_folds = 5
skf = list(cross_validation.StratifiedKFold(y, n_folds))

# http://scikit-learn.org/dev/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
# http://scikit-learn.org/stable/modules/calibration.html
clfs = [KNeighborsClassifier(n_jobs=-1),
        ensemble.BaggingClassifier(n_jobs=-1),
        ensemble.AdaBoostClassifier(),
        svm.SVC(probability=True, kernel='rbf', C=.01),
        neural_network.MLPClassifier(algorithm='l-bfgs', early_stopping=True)
]

blend_train = numpy.zeros((X.shape[0], len(clfs)))
blend_test = numpy.zeros((predict_X.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    #print clf.__class__
    blend_test_j = numpy.zeros((predict_X.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        #print 'fold', i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        predict_y = clf.predict_proba(X_test)[:,1]
        blend_train[test, j] = predict_y
        blend_test_j[:, i] = clf.predict_proba(predict_X)[:,1]
    blend_test[:,j] = blend_test_j.mean(1)

clf = LogisticRegression()
clf.fit(blend_train, y)
predict_y = clf.predict_proba(blend_test)[:,1]

pmatchups = possible_tourney_matchups(predict_seasons)

#print score_predictions(pmatchups, predict_y)

write_predictions(pmatchups, predict_y)

print(datetime.datetime.now())
