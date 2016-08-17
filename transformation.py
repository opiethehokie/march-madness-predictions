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
import operator
import datetime
import pandas
import math

import constants

from collections import defaultdict

from statistics import rpi_and_sos
from statistics import pythagorean_expectations
from statistics import score_stddevs
from statistics import avg_per_game
from util import possible_tourney_matchups
from numpy import NaN


#def tourney_seeds():
#    seeds = defaultdict(lambda: defaultdict(int))
#    with open('data/tourney_seeds.csv') as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            season = int(row['Season'])
#            seed = row['Seed']
#            team = int(row['Team'])
#            seeds[team][season] = seed
#    return seeds

def save(known, unknown):
    with open("data/train.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(known)
    with open("data/predict.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(unknown)

def all_teams():
    teams = defaultdict(set)
    with open('data/regular_season_detailed_results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['Season'])
            teams[season].add(int(row['Wteam']))
            teams[season].add(int(row['Lteam']))
    return teams

teams = all_teams()

games = defaultdict(lambda: defaultdict(int))
opponents = defaultdict(lambda: defaultdict(lambda: []))
h2h = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0]*2)))
diffs = defaultdict(lambda: defaultdict(lambda: []))
results = defaultdict(lambda: defaultdict(lambda: defaultdict(int, {'homewins': 0, 'awaywins': 0, 'neutralwins': 0, 'otwins': 0, 'totalwins': 0,
                                                                    'homelosses': 0, 'awaylosses': 0, 'neutrallosses': 0, 'otlosses': 0, 'totallosses': 0})))
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float, {'adj-wins': 0})))
rpis = soss = {}

def accumulate_results(row):
    wteam = int(row['Wteam'])
    lteam = int(row['Lteam'])
    season = int(row['Season'])
    if row['Wloc'] == 'H':
        results[wteam][season]['homewins'] += 1
        results[lteam][season]['awaylosses'] += 1
        results[wteam][season]['for-points'] += int(row['Wscore']) - constants.HOME_COURT_ADVANTAGE
        results[wteam][season]['against-points'] += int(row['Lscore']) + constants.HOME_COURT_ADVANTAGE
        results[lteam][season]['for-points'] += int(row['Lscore']) + constants.HOME_COURT_ADVANTAGE
        results[lteam][season]['against-points'] += int(row['Wscore']) - constants.HOME_COURT_ADVANTAGE
    elif row['Wloc'] == 'N':
        results[wteam][season]['neutralwins'] += 1
        results[lteam][season]['neutrallosses'] += 1
        results[wteam][season]['for-points'] += int(row['Wscore'])
        results[wteam][season]['against-points'] += int(row['Lscore'])
        results[lteam][season]['for-points'] += int(row['Lscore'])
        results[lteam][season]['against-points'] += int(row['Wscore'])
    elif row['Wloc'] == 'A':
        results[wteam][season]['awaywins'] += 1
        results[lteam][season]['homelosses'] += 1
        results[wteam][season]['for-points'] += int(row['Wscore']) + constants.HOME_COURT_ADVANTAGE
        results[wteam][season]['against-points'] += int(row['Lscore']) - constants.HOME_COURT_ADVANTAGE
        results[lteam][season]['for-points'] += int(row['Lscore']) - constants.HOME_COURT_ADVANTAGE
        results[lteam][season]['against-points'] += int(row['Wscore']) + constants.HOME_COURT_ADVANTAGE
    if int(row['Numot']) > 0:
        results[wteam][season]['otwins'] += 1
        results[lteam][season]['otlosses'] += 1
    results[wteam][season]['totalwins'] += 1
    results[lteam][season]['totallosses'] += 1

def accumulate_stats(row):
    wteam = int(row['Wteam'])
    lteam = int(row['Lteam'])
    season = int(row['Season'])
    wfgm = float(row['Wfgm'])
    wfga = float(row['Wfga'])
    wfgm3 = float(row['Wfgm3'])
    wfga3 = float(row['Wfga3'])
    wfta = float(row['Wfta'])
    wftm = float(row['Wftm'])
    wor = float(row['Wor'])
    wdr = float(row['Wdr'])
    wast = float(row['Wast'])
    wto = float(row['Wto']) 
    wstl = float(row['Wstl'])
    wblk = float(row['Wblk'])
    wscore = float(row['Wscore'])
    wpf = float(row['Wpf'])
    lfgm = float(row['Lfgm'])
    lfga = float(row['Lfga'])
    lfgm3 = float(row['Lfgm3'])
    lfga3 = float(row['Lfga3'])
    lfta = float(row['Lfta'])
    lftm = float(row['Lftm'])
    lor = float(row['Lor'])
    ldr = float(row['Ldr'])
    last = float(row['Last'])
    lto = float(row['Lto']) 
    lstl = float(row['Lstl'])
    lblk = float(row['Lblk'])
    lscore = float(row['Lscore'])
    lpf = float(row['Lpf'])
    # possessions
    wposs = wfga - wor + wto + .475 * wfta
    lposs = lfga - lor + lto + .475 * lfta
    # effective field goal percentage
    stats[wteam][season]['oefgp'] += (wfgm + .5 * wfgm3) / wfga
    stats[lteam][season]['oefgp'] += (lfgm + .5 * lfgm3) / lfga
    stats[lteam][season]['defgp'] += (wfgm + .5 * wfgm3) / wfga
    stats[wteam][season]['defgp'] += (lfgm + .5 * lfgm3) / lfga
    # rebounding percentages
    stats[wteam][season]['orbp'] += wor / (wor + ldr)
    stats[lteam][season]['orbp'] += lor / (lor + wdr)
    stats[lteam][season]['drbp'] += ldr / (ldr + wor)
    stats[wteam][season]['drbp'] += wdr / (wdr + lor)
    # free throw rate
    stats[wteam][season]['oftr'] += wfta / wfga
    stats[lteam][season]['oftr'] += lfta / lfga
    stats[lteam][season]['dftr'] += wfta / wfga
    stats[wteam][season]['dftr'] += lfta / lfga
    # turnover percentage
    stats[wteam][season]['top'] += wto / wposs
    stats[lteam][season]['top'] += lto / lposs
    # off efficiency
    stats[wteam][season]['oeff'] += wscore / wposs
    stats[lteam][season]['oeff'] += lscore / lposs
    stats[lteam][season]['deff'] += wscore / wposs
    stats[wteam][season]['deff'] += lscore / lposs
    # assist rate
    stats[wteam][season]['ar'] += wast / wfgm
    stats[lteam][season]['ar'] += last / lfgm
    # block rate
    stats[wteam][season]['br'] += wblk / lfga
    stats[lteam][season]['br'] += lblk / wfga
    # steal rate
    stats[wteam][season]['sr'] += wstl / lposs
    stats[lteam][season]['sr'] += lstl / wposs
    # performance index rating
    stats[wteam][season]['pir'] += ((wscore + wor + wdr + wast + wstl + wblk + lpf) - ((wfga - wfgm) + (wfga3 - wfgm3) + (wfta - wftm) + wto + lblk + wpf))
    stats[lteam][season]['pir'] += ((lscore + lor + ldr + last + lstl + lblk + wpf) - ((lfga - lfgm) + (lfga3 - lfgm3) + (lfta - lftm) + lto + wblk + lpf))
    
    stats[wteam][season]['adj-wins'] += math.log10(int(row['Daynum'])+1)

def accumulate_info(row):
    season = int(row['Season'])
    wteam = int(row['Wteam'])
    lteam = int(row['Lteam'])
    wscore = int(row['Wscore'])
    lscore = int(row['Lscore'])
    games[wteam][season] += 1
    games[lteam][season] += 1
    opponents[wteam][season].append(lteam)
    opponents[lteam][season].append(wteam)
    diffs[wteam][season].append(min(wscore - lscore, constants.MAX_SCORE_MARGIN)) 
    diffs[lteam][season].append(max(lscore - wscore, -constants.MAX_SCORE_MARGIN))
    h2h[wteam][season][lteam] = map(operator.add, h2h[wteam][season][lteam], [1, 0])
    h2h[lteam][season][wteam] = map(operator.add, h2h[lteam][season][wteam], [0, 1])

def arg_vector(team, season):
    rpi = rpis[team][season]
    sos = soss[team][season]
    pythag = pythagorean_expectations(results[team][season], games[team][season])
    consistency = score_stddevs(diffs[team][season])
    avg_stats = avg_per_game(stats[team][season], games[team][season])
    return [rpi, sos, pythag, consistency] + avg_stats

def good_game(rpi1, rpi2):
    return rpi1 > constants.MIN_RPI and rpi2 > constants.MIN_RPI

def transform_known_games(results_file):
    global rpis, soss
    train = []
    current_day = 0
    df = pandas.read_csv(results_file)
    df = df.sort_values(by = ['Daynum'])
    for _, row in df.iterrows():
        season = int(row['Season'])
        wteam = int(row['Wteam'])
        lteam = int(row['Lteam'])
        day = int(row['Daynum'])
        if day < constants.TOURNEY_START_DAY:
            accumulate_info(row)
            accumulate_results(row)
            accumulate_stats(row)
            if day > current_day and day > constants.REAL_SEASON_START_DAY:
                rpis, soss = rpi_and_sos(results, opponents, h2h)
                current_day = day
        if day > constants.REAL_SEASON_START_DAY:
            if good_game(rpis[wteam][season], rpis[lteam][season]):
                wvector = arg_vector(wteam, season)
                lvector = arg_vector(lteam, season)
                if wteam < lteam:
                    train.append([season, day, 1] + wvector + lvector)
                else:
                    train.append([season, day, 0] + lvector + wvector)
    return train

def transform_unknown_games():
    predict = []
    pmatchups = possible_tourney_matchups() 
    for pmatchup in pmatchups:
        season, teama, teamb = map(int, pmatchup.split('_'))
        avector = arg_vector(teama, season)
        bvector = arg_vector(teamb, season)
        predict.append([season, NaN, NaN] + avector + bvector) 
    return predict

print(datetime.datetime.now())
            
train = transform_known_games('data/regular_season_detailed_results.csv') + transform_known_games('data/tourney_detailed_results.csv')
predict = transform_unknown_games()

save(train, predict)

print(datetime.datetime.now())
