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

import pandas
import csv

def prune_rankings():
    old_rankings = pandas.read_csv('~/Downloads/massey_ordinals_2003-2015.csv')
    current_rankings = pandas.read_csv('~/Downloads/MasseyOrdinals2016ThruDay114.csv')
    rankings = pandas.concat([old_rankings, current_rankings], ignore_index=True)
    colley_rankings = rankings[rankings['sys_name'] == 'COL'].drop('sys_name', axis=1).rename(columns={ 'orank' : 'colley'})
    massey_rankings = rankings[rankings['sys_name'] == 'MAS'].drop('sys_name', axis=1).rename(columns={ 'orank' : 'massey'})
    rankings = pandas.merge(colley_rankings, massey_rankings, how='outer')
    rankings.to_csv('data/rankings.csv')
    
#def possible_tourney_matchups(seasons = None):
#    teams = defaultdict(set)
#    with open('data/regular_season_detailed_results.csv') as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            season = int(row['Season'])
#            teams[season].add(int(row['Wteam']))
#            teams[season].add(int(row['Lteam']))
#    seasons = seasons or teams.keys()
#    matchups = []
#    for season in seasons:
#        seasons_teams = teams[season]
#        for team1 in seasons_teams:
#            for team2 in seasons_teams:
#                if team1 < team2:
#                    matchups.append(str(season) + '_' + str(team1) + '_' + str(team2))
#    return matchups

def possible_tourney_matchups(seasons = None):
    matchups = []
    with open('submissions/sample_submission.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season, _, _ = map(int, row['Id'].split('_'))
            if not seasons or season in seasons:
                matchups.append(row['Id'])
    return matchups

#def possible_tourney_matchups(seasons = None):
#    teams = defaultdict(set)
#    with open('data/tourney_detailed_results.csv') as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            season = int(row['Season'])
#            teams[season].add(int(row['Wteam']))
#            teams[season].add(int(row['Lteam']))
#    seasons = seasons or teams.keys()
#    matchups = []
#    for season in seasons:
#        seasons_teams = teams[season]
#        for team1 in seasons_teams:
#            for team2 in seasons_teams:
#                if team1 < team2:
#                    matchups.append(str(season) + '_' + str(team1) + '_' + str(team2))
#    return matchups
