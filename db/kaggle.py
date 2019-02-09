#   Copyright 2016-2019 Michael Peters
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


from collections import defaultdict

import csv
import numpy as np
import pandas as pd


def read_raw_data(season_file, tourney_file):
    return pd.concat([pd.read_csv(season_file), pd.read_csv(tourney_file)]).sort_values(by=['Daynum', 'Wteam', 'Lteam'])

def read_predictions(predict_file):
    with open(predict_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        predictions = {row['Id'][5:]: float(row['Pred']) for row in reader}
        return predictions

def write_predictions(matchups, predictions, predict_file):
    with open(predict_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['Id', 'Pred'])
        writer.writerows(np.column_stack((matchups, predictions)))

def possible_tourney_matchups(sample_submission_file):
    matchups = pd.read_csv(sample_submission_file)['Id']
    fake_boxscores = []
    for matchup in matchups:
        season, teama, teamb = map(int, matchup.split('_'))
        fake_boxscores.append([season, 999, teama, teamb])
    return matchups, pd.DataFrame(fake_boxscores, columns=['Season', 'Daynum', 'Wteam', 'Lteam'])

def team_id_mapping(team_file):
    with open(team_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        teams = {row['Team_Name']: int(row['Team_Id']) for row in reader}
    return teams

def team_seed_mapping(seed_file):
    seeds = defaultdict(lambda: defaultdict(str))
    with open(seed_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seeds[row['Season']][row['Team']] = row['Seed']
    return seeds

def championship_pairings(slot_file):
    slots = defaultdict()
    with open(slot_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Slot'] == 'R6CH':
                slots[row['Season']] = (row['Strongseed'], row['Weakseed'])
    return slots
