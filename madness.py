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
import numpy
import pandas
import yaml

import yamlordereddictloader

from sklearn.externals import joblib

from ml.classification import train_stacked_model
from ml.simulations import simulate_tourney
from ml.wrangling import custom_train_test_split, oversample_tourney_games


numpy.random.seed(42) # helps get similar results for everyone run

TOURNEY_DATA_FILE = 'data/tourney_detailed_results_2015.csv'
SEASON_DATA_FILE = 'data/regular_season_detailed_results_2016.csv'
SAMPLE_SUBMISSION_FILE = 'results/sample_submission_2015.csv'
SUBMISSION_FILE = 'results/submission.csv'
TEAMS_FILE = 'data/teams.csv'
TOURNEY_FORMAT_FILE = 'data/tourney_format_2015.yml'
PERSISTED_MODEL_FILE = 'results/stacked-model.pkl'


def clean_raw_data(start_year=2009, start_day=25):
    def read_data(results_file):
        return pandas.read_csv(results_file)           
    data = pandas.concat([read_data(SEASON_DATA_FILE), read_data(TOURNEY_DATA_FILE)]).sort_values(by='Daynum')
    preseason = data.pipe(lambda df: df[df.Season >= start_year]).pipe(lambda df: df[df.Daynum < start_day])
    season = data.pipe(lambda df: df[df.Season >= start_year]).pipe(lambda df: df[df.Daynum >= start_day])
    return preseason, season

def write_predictions(matchups, predictions):
    with open(SUBMISSION_FILE, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Pred'])
        writer.writerows(numpy.column_stack((matchups, predictions)))

def read_predictions():
    with open(SUBMISSION_FILE) as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        predictions = { k[5:]:float(v) for k, v in reader }
    return predictions

def persist_model(m):
    joblib.dump(m.best_estimator_, PERSISTED_MODEL_FILE)

def load_persisted_model():
    return joblib.load(PERSISTED_MODEL_FILE)

def possible_tourney_matchups():
    matchups = pandas.read_csv(SAMPLE_SUBMISSION_FILE)['Id']
    fake_boxscores = []
    for matchup in matchups:
        season, teama, teamb = map(int, matchup.split('_'))
        fake_boxscores.append([season, 137, teama, teamb])
    return matchups, pandas.DataFrame(fake_boxscores, columns=['Season', 'Daynum', 'Wteam', 'Lteam'])

def team_id_mapping():
    with open(TEAMS_FILE) as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        teams = { v:int(k) for k, v in reader }
    return teams


preseason_games, games = clean_raw_data(start_year=2013, start_day=50)
games = oversample_tourney_games(games) #TODO

X_train, X_test, y_train, y_test = custom_train_test_split(games, 2015)
model = train_stacked_model(preseason_games, X_train, X_test, y_train, y_test)

persist_model(model)
model = load_persisted_model()

predict_matchups, X_predict = possible_tourney_matchups()
y_predict = model.predict_proba(X_predict)[:,1]
write_predictions(predict_matchups, y_predict)
    
simulate_tourney(team_id_mapping(), read_predictions(), yaml.load(open(TOURNEY_FORMAT_FILE), Loader=yamlordereddictloader.Loader))
