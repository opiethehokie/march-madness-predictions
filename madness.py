#   Copyright 2016-2018 Michael Peters
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


import os
import sys

from collections import defaultdict

import csv
import numpy
import pandas
import yaml
import yamlordereddictloader

from sklearn.metrics import classification_report, log_loss, accuracy_score

from ml.predictions import train_model
from ml.simulations import simulate_tourney
from ml.wrangling import (custom_train_test_split, filter_outlier_games, adjust_overtime_games,
                          assemble_features, create_synthetic_games)


TOURNEY_DATA_FILE = 'data/tourney_detailed_results_2017.csv'
SEASON_DATA_FILE = 'data/regular_season_detailed_results_2018.csv'
SUBMISSION_FILE = 'results/submission.csv'
TEAMS_FILE = 'data/teams.csv'
SEEDS_FILE = 'data/seeds_2018.csv'
SLOTS_FILE = 'data/slots_2018.csv'
FEATURE_CACHE_FILE = 'data/feature_cache.csv'
PREDICT_CACHE_FILE = 'data/predict_cache.csv'


def clean_raw_data(syear, sday, eyear):
    def read_data(results_file):
        return pandas.read_csv(results_file)
    data = (pandas.concat([read_data(SEASON_DATA_FILE), read_data(TOURNEY_DATA_FILE)])
            .sort_values(by='Daynum'))
    preseason = (data.pipe(lambda df: df[df.Season >= syear])
                 .pipe(lambda df: df[df.Season <= eyear])
                 .pipe(lambda df: df[df.Daynum < sday]))
    season = (data.pipe(lambda df: df[df.Season >= syear])
              .pipe(lambda df: df[df.Season <= eyear])
              .pipe(lambda df: df[df.Daynum >= sday]))
    assert numpy.all(preseason >= 0)
    assert numpy.all(season >= 0)
    return preseason, season

def write_predictions(matchups, predictions, suffix=''):
    with open(SUBMISSION_FILE.replace('.csv', suffix + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['Id', 'Pred'])
        writer.writerows(numpy.column_stack((matchups, predictions)))

def differentiate_final_predictions(matchups, predictions, new_value):
    diff_predictions = list(predictions)
    slots = championship_pairings()
    seeds = team_seed_mapping()
    for idx, matchup in enumerate(matchups):
        if possible_tourney_final(slots, seeds, matchup):
            diff_predictions[idx] = new_value
        else:
            if diff_predictions[idx] >= .7:
                diff_predictions[idx] = diff_predictions[idx] + .001
            if diff_predictions[idx] <= .3:
                diff_predictions[idx] = diff_predictions[idx] - .001
            assert diff_predictions[idx] > 0 and diff_predictions[idx] < 1
    return diff_predictions

def read_predictions():
    with open(SUBMISSION_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        predictions = {row['Id'][5:]: float(row['Pred']) for row in reader}
        return predictions

def possible_tourney_matchups():
    matchups = pandas.read_csv(SAMPLE_SUBMISSION_FILE)['Id']
    fake_boxscores = []
    for matchup in matchups:
        season, teama, teamb = map(int, matchup.split('_'))
        fake_boxscores.append([season, 137, teama, teamb])
    return matchups, pandas.DataFrame(fake_boxscores, columns=['Season', 'Daynum', 'Wteam', 'Lteam'])

def possible_tourney_final(slots, seeds, matchup):
    year, teama, teamb = matchup.split('_')
    teama_region = seeds[year][teama][0]
    teamb_region = seeds[year][teamb][0]
    (champ_regions1, champ_regions2) = slots[year]
    return ((champ_regions1.find(teama_region) > -1 and champ_regions2.find(teamb_region) > -1) or
            (champ_regions2.find(teama_region) > -1 and champ_regions1.find(teamb_region) > -1))

def team_id_mapping():
    with open(TEAMS_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        teams = {row['Team_Name']: int(row['Team_Id']) for row in reader}
    return teams

def team_seed_mapping():
    seeds = defaultdict(lambda: defaultdict(str))
    with open(SEEDS_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seeds[row['Season']][row['Team']] = row['Seed']
    return seeds

def championship_pairings():
    slots = defaultdict()
    with open(SLOTS_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Slot'] == 'R6CH':
                slots[row['Season']] = (row['Strongseed'], row['Weakseed'])
    return slots

def add_features(pre_data, data, post_data):
    if not os.path.isfile(FEATURE_CACHE_FILE) or not os.path.isfile(PREDICT_CACHE_FILE):
        features, features_predict = assemble_features(pre_data, data, post_data)
        features.to_csv(FEATURE_CACHE_FILE)
        pandas.DataFrame(features_predict).to_csv(PREDICT_CACHE_FILE)
        assert features.shape[1] == 34 + 1108
        assert features_predict.shape[1] == 4 + 1108
    features = pandas.read_csv(FEATURE_CACHE_FILE, index_col=0)
    features_predict = pandas.read_csv(PREDICT_CACHE_FILE, index_col=0)
    return features, features_predict.values.astype('float64')


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2017

    SAMPLE_SUBMISSION_FILE = 'results/sample_submission_%s.csv' % predict_year
    TOURNEY_FORMAT_FILE = 'data/tourney_format_%s.yml' % predict_year

    start_year = predict_year - 4
    start_day = 30

    predict_matchups, postseason_games = possible_tourney_matchups()
    preseason_games, games = clean_raw_data(start_year, start_day, predict_year)

    games = adjust_overtime_games(games)
    games = filter_outlier_games(games)
    games = pandas.concat([games, create_synthetic_games(games)])

    games, X_predict = add_features(preseason_games, games, postseason_games)

    X_train, X_test, y_train, y_test = custom_train_test_split(games, predict_year)
    model = train_model(X_train, y_train)

    if X_test.size > 0:
        y_predict = model.predict(X_test)
        print(classification_report(y_test, y_predict))
        print('Accuracy is %f' % accuracy_score(y_test, y_predict))
        y_predict_probas = model.predict_proba(X_test)
        print('Log loss is %f' % log_loss(y_test, y_predict_probas))

    y_predict = model.predict_proba(X_predict)[:, 1]
    write_predictions(predict_matchups, y_predict)

    # post-processing for Kaggle competition (two submissions means we can always get championship game correct)
    write_predictions(predict_matchups, differentiate_final_predictions(predict_matchups, y_predict, 0), '0')
    write_predictions(predict_matchups, differentiate_final_predictions(predict_matchups, y_predict, 1), '1')

    # predict actual tournament bracket for cash money
    if predict_year >= 2015:
        simulate_tourney(team_id_mapping(), read_predictions(), yaml.load(open(TOURNEY_FORMAT_FILE), Loader=yamlordereddictloader.Loader))
