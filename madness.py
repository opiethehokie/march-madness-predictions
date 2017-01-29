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
import sys
import yaml

import yamlordereddictloader

from collections import defaultdict

from ml.classification import train_stacked_model
from ml.simulations import simulate_tourney
from ml.wrangling import custom_train_test_split, filter_outlier_games, oversample_tourney_games


numpy.random.seed(42) # helps get repeatable results

TOURNEY_DATA_FILE = 'data/tourney_detailed_results_2015.csv'
SEASON_DATA_FILE = 'data/regular_season_detailed_results_2016.csv'
SUBMISSION_FILE = 'results/submission.csv'
TEAMS_FILE = 'data/teams.csv'
SEEDS_FILE = 'data/seeds_2016.csv'
SLOTS_FILE = 'data/slots_2016.csv'


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
    return preseason, season

def write_predictions(matchups, predictions, suffix=''):
    with open(SUBMISSION_FILE.replace('.csv', suffix + '.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Pred'])
        writer.writerows(numpy.column_stack((matchups, predictions)))

def differentiate_final_predictions(matchups, predictions, new_value):
    diff_predictions = list(predictions)
    slots = championship_pairings()
    seeds = team_seed_mapping()
    for idx, matchup in enumerate(matchups):
        if possible_tourney_final(slots, seeds, matchup):
            diff_predictions[idx] = new_value
    return diff_predictions

def read_predictions():
    with open(SUBMISSION_FILE) as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        predictions = { k[5:]:float(v) for k, v in reader }
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
    with open(TEAMS_FILE) as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        teams = { v:int(k) for k, v in reader }
    return teams

def team_seed_mapping():
    seeds = defaultdict(lambda: defaultdict(str))
    with open(SEEDS_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seeds[row['Season']][row['Team']] = row['Seed']
    return seeds

def championship_pairings():
    slots = defaultdict()
    with open(SLOTS_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Slot'] == 'R6CH':
                slots[row['Season']] = (row['Strongseed'], row['Weakseed'])
    return slots


# base hyperparameters

predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2015

start_year = predict_year - 3
start_day = 30

SAMPLE_SUBMISSION_FILE = 'results/sample_submission_%s.csv' % predict_year
TOURNEY_FORMAT_FILE = 'data/tourney_format_%s.yml' % predict_year

outlier_std_devs = 6
tourney_multiplyer = 5

# initial pre-processing
preseason_games, games = clean_raw_data(start_year, start_day, predict_year)
games = filter_outlier_games(games, outlier_std_devs)

_, X_test, _, y_test = custom_train_test_split(games, predict_year)
games = oversample_tourney_games(games, tourney_multiplyer)

# training
X_train, _, y_train, _ = custom_train_test_split(games, predict_year)
model = train_stacked_model(preseason_games, X_train, X_test, y_train, y_test)

if predict_year >= 2015:

    # predict all possible tournament games for Kaggle competition
    predict_matchups, X_predict = possible_tourney_matchups()
    y_predict = model.predict_proba(X_predict)[:,1]
    write_predictions(predict_matchups, y_predict)

    # post-processing for Kaggle competition (two submissions means we can always get championship game correct)
    write_predictions(predict_matchups, differentiate_final_predictions(predict_matchups, y_predict, 0), '0')
    write_predictions(predict_matchups, differentiate_final_predictions(predict_matchups, y_predict, 1), '1')

    # predict actual tournament bracket
    simulate_tourney(team_id_mapping(), read_predictions(), yaml.load(open(TOURNEY_FORMAT_FILE), Loader=yamlordereddictloader.Loader))
