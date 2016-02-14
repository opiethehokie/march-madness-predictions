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
#   limitations under the License.]

import numpy

import constants

from collections import defaultdict
from sklearn import metrics


def rpi_and_sos(results, opponents, h2h):
    rpi = defaultdict(lambda: defaultdict(float))
    sos = defaultdict(lambda: defaultdict(float))
    wps = _winning_percentages(results)
    owps = _opponents_avg_winning_percentages(opponents, results, h2h)
    oowps = _opponents_opponents_avg_winning_percentages(opponents, owps)
    for team in results.keys():
        for season in results[team].keys():
            wp = wps[team][season]
            owp = owps[team][season]
            oowp = oowps[team][season]         
            rpi[team][season] = (0.25)*(wp) + (0.50)*(owp) + (0.25)*(oowp)
            sos[team][season] = (2.0/3.0 * owp) + (1.0/3.0 * oowp)
    return rpi, sos

def _winning_percentages(results):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in results.keys():
        for season in results[team].keys():
            home_wins = results[team][season]['homewins']
            neutral_wins = results[team][season]['neutralwins']
            road_wins = results[team][season]['awaywins']
            home_losses = results[team][season]['homelosses']
            neutral_losses = results[team][season]['neutrallosses']
            road_losses = results[team][season]['awaylosses']
            wins = 0.6 * home_wins + 1.0 * neutral_wins + 1.4 * road_wins
            losses = 1.4 * home_losses + 1.0 * neutral_losses + 0.6 * road_losses
            percentages[team][season] = wins / (wins + losses) if (wins + losses) > 0 else 0
    return percentages

def _opponents_avg_winning_percentages(opponents, results, h2h):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in opponents.keys():
        for season in opponents[team].keys():
            num_opponents = len(opponents[team][season])
            i = 0
            for o in opponents[team][season]:
                wins = results[o][season]['homewins'] + results[o][season]['neutralwins'] + results[o][season]['awaywins'] - h2h[team][season][o][1]
                losses = results[o][season]['homelosses'] + results[o][season]['neutrallosses'] + results[o][season]['awaylosses'] - h2h[team][season][o][0]
                percentages[team][season] += 1.0 * wins / (wins + losses) if (wins + losses) > 0 else 0
                i +=1 
            percentages[team][season] = percentages[team][season] / num_opponents
    return percentages

def _opponents_opponents_avg_winning_percentages(opponents, opponents_winning_percentages):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in opponents.keys():
        for season in opponents[team].keys():
            for o in opponents[team][season]:
                percentages[team][season] += opponents_winning_percentages[o][season]
            percentages[team][season] = percentages[team][season] / len(opponents[team][season])
    return percentages

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
def logloss(act, pred):
    return metrics.log_loss(act, pred)

def pythagorean_expectations(results, num_games):
    exp = constants.PYTHAGOREAN_EXPECTATION_EXP
    points_for = results['for-points']
    points_against = results['against-points']
    win_percent = float(results['totalwins']) / num_games
    return win_percent - float(points_for)**exp / (float(points_for)**exp + float(points_against)**exp)

def score_stddevs(diffs):
    return numpy.std(diffs) if len(diffs) > 0 else 0

def avg_per_game(stats, num_games):
    sorted_stats = dict(sorted(stats.items(), key = lambda x :x[0]))
    #print sorted_stats.keys()
    return map(lambda x: x / float(num_games), sorted_stats.values())
