#   Copyright 2015 Michael Peters
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
import numpy

from collections import defaultdict

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def team_id_mapping():
    with open('data/teams.csv') as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        teams = { int(k):v for k,v in reader }
    return teams

def reg_season_stats():
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    opponents = defaultdict(lambda: defaultdict(lambda: []))
    h2h = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0]*2))) # win/loss against specific opponent
    with open('data/regular_season_detailed_results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:            
            season = int(row['season'])
            daynum = int(row['daynum'])
            wteam = int(row['wteam'])
            lteam = int(row['lteam'])
            wloc = row['wloc']
            
            if daynum < 25:
                pass
            
            stats[wteam][season]['games'] += 1
            stats[lteam][season]['games'] += 1
            
            if wloc == 'H':
                stats[wteam][season]['homewins'] += 1
                stats[lteam][season]['awaylosses'] += 1
            elif wloc == 'N':
                stats[wteam][season]['neutralwins'] += 1
                stats[lteam][season]['neutrallosses'] += 1
            elif wloc == 'A':
                stats[wteam][season]['awaywins'] += 1
                stats[lteam][season]['homelosses'] += 1
                
            if int(row['numot']) > 0:
                stats[wteam][season]['otwins'] += 1
                stats[lteam][season]['otlosses'] += 1
            
            stats[wteam][season]['fgm'] += int(row['wfgm'])
            stats[wteam][season]['fga'] += int(row['wfga'])
            stats[wteam][season]['fgm3'] += int(row['wfgm3'])
            stats[wteam][season]['fga3'] += int(row['wfga3'])
            stats[wteam][season]['ftm'] += int(row['wftm'])
            stats[wteam][season]['fta'] += int(row['wfta'])
            stats[wteam][season]['or'] += int(row['wor'])
            stats[wteam][season]['dr'] += int(row['wdr'])
            stats[wteam][season]['ast'] += int(row['wast'])
            stats[wteam][season]['to'] += int(row['wto'])
            stats[wteam][season]['stl'] += int(row['wstl'])
            stats[wteam][season]['blk'] += int(row['wblk'])
            stats[wteam][season]['pf'] += int(row['wpf'])
            
            stats[lteam][season]['fgm'] += int(row['lfgm'])
            stats[lteam][season]['fga'] += int(row['lfga'])
            stats[lteam][season]['fgm3'] += int(row['lfgm3'])
            stats[lteam][season]['fga3'] += int(row['lfga3'])
            stats[lteam][season]['ftm'] += int(row['lftm'])
            stats[lteam][season]['fta'] += int(row['lfta'])
            stats[lteam][season]['or'] += int(row['lor'])
            stats[lteam][season]['dr'] += int(row['ldr'])
            stats[lteam][season]['ast'] += int(row['last'])
            stats[lteam][season]['to'] += int(row['lto'])
            stats[lteam][season]['stl'] += int(row['lstl'])
            stats[lteam][season]['blk'] += int(row['lblk'])
            stats[lteam][season]['pf'] += int(row['lpf'])

            opponents[wteam][season].append(lteam)
            opponents[lteam][season].append(wteam) 
                          
            h2h[wteam][season][lteam] = map(operator.add, h2h[wteam][season][lteam], [1, 0])
            h2h[lteam][season][wteam] = map(operator.add, h2h[lteam][season][wteam], [0, 1])
            
    return stats, opponents, h2h

def winning_percentages(stats):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in stats.keys():
        for season in stats[team].keys():
            home_wins = stats[team][season]['homewins']
            neutral_wins = stats[team][season]['neutralwins']
            road_wins = stats[team][season]['awaywins']
            home_losses = stats[team][season]['homelosses']
            neutral_losses = stats[team][season]['neutrallosses']
            road_losses = stats[team][season]['awaylosses']
            wins = 0.6 * home_wins + 1.0 * neutral_wins + 1.4 * road_wins
            losses = 1.4 * home_losses + 1.0 * neutral_losses + 0.6 * road_losses
            percentages[team][season] = wins / (wins + losses)
    return percentages

def opponents_avg_winning_percentages(opponents, stats, h2h):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in opponents.keys():
        for season in opponents[team].keys():
            num_opponents = len(opponents[team][season])
            i = 0
            for o in opponents[team][season]:
                wins = stats[o][season]['homewins'] + stats[o][season]['neutralwins'] + stats[o][season]['awaywins'] - h2h[team][season][o][1]
                losses = stats[o][season]['homelosses'] + stats[o][season]['neutrallosses'] + stats[o][season]['awaylosses'] - h2h[team][season][o][0]
                percentages[team][season] += 1.0 * wins / (wins + losses)
                i +=1 
            percentages[team][season] = percentages[team][season] / num_opponents
    return percentages

def opponents_opponents_avg_winning_percentages(opponents, opponents_winning_percentages):
    percentages = defaultdict(lambda: defaultdict(float))
    for team in opponents.keys():
        for season in opponents[team].keys():
            for o in opponents[team][season]:
                percentages[team][season] += opponents_winning_percentages[o][season]
            percentages[team][season] = percentages[team][season] / len(opponents[team][season])
    return percentages

def rpi_and_sos(stats, opponents, h2h):
    rpi = defaultdict(lambda: defaultdict(float))
    sos = defaultdict(lambda: defaultdict(float))
    wps = winning_percentages(stats)
    owps = opponents_avg_winning_percentages(opponents, stats, h2h)
    oowps = opponents_opponents_avg_winning_percentages(opponents, owps)
    for team in stats.keys():
        for season in stats[team].keys():
            wp = wps[team][season]
            owp = owps[team][season]
            oowp = oowps[team][season]         
            rpi[team][season] = (0.25)*(wp) + (0.50)*(owp) + (0.25)*(oowp)
            sos[team][season] = (2.0/3.0 * owp) + (1.0/3.0 * oowp)
    return rpi, sos

def tourney_seeds():
    seeds = defaultdict(lambda: defaultdict(int))
    with open('data/tourney_seeds.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['season'])
            seed = row['seed']
            team = int(row['team'])
            seeds[team][season] = seed
    return seeds

def season_ranks():
    #fieldnames = ['season','rating_day_num','sys_name','team','orank']
    #reader = csv.DictReader(open(r"data/massey_ordinals.csv"))
    #filtered = filter(lambda p: 'WLK' == p['sys_name'] and '133' == p['rating_day_num'], reader)
    #writer = csv.DictWriter(open(r"data/filtered_massey_ordinals.csv", 'wb'), fieldnames=fieldnames)
    #writer.writeheader()
    #writer.writerows(filtered)
    ranks = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with open('data/filtered_massey_ordinals.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['season'])
            team = int(row['team'])
            rank = int(row['orank']) 
            # WLK - http://sawhitlock.home.mindspring.com/fbrankdesc.htm
            # MOR also pretty good - http://sonnymoorepowerratings.com/moore.htm
            ranks[team][season] = rank
    return ranks

def slot(s1, s2):
    first = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
    second = [(1,8),(1,9),(16,8),(16,9),(4,5),(4,12),(13,5),(13,12),(3,6),(3,11),(14,6),(14,11),(2,7),(2,10),(15,7),15,10]
    third = [(1,4),(1,5),(1,12),(1,13),(8,4),(8,5),(8,12),(8,13),(9,4),(9,5),(9,12),(9,13),(16,4),(16,5),(16,12),(16,13),
             (3,2),(3,7),(3,10),(3,15),(6,2),(6,7),(6,10),(6,15),(11,2),(11,7),(11,10),(11,15),(14,2),(14,7),(14,10),(14,15)]
    r1 = s1[0]
    r2 = s2[0]
    n1 = int(s1[1:3])
    n2 = int(s2[1:3])
    if r1 == r2:
        if n1 == n2:
            return 0
        if (n1,n2) in first or (n2,n1) in first:
            return 1
        if (n1,n2) in second or (n2,n1) in second:
            return 2
        if (n1,n2) in third or (n2,n1) in third:
            return 3
        return 4
    return 5

def avg_stats(stats):    
    games = stats['games']
    ftm = stats['ftm']
    fta = stats['fta']
    dreb = stats['dr']
    stl = stats['stl']
    blk = stats['blk']
    pf = stats['pf']
    otwins = stats['otwins']
    otlosses = stats['otlosses']
    statv = [otwins, otlosses, games, float(ftm)/float(fta), pf, blk, stl, dreb]
    return map(lambda x: x / float(games), statv)
            
def data(pseasons, pmatchups):
    stats, opponents, h2h = reg_season_stats()
    rpi, _ = rpi_and_sos(stats, opponents, h2h)
    seeds = tourney_seeds()
    ranks = season_ranks()
    games = []
    results = []
    predict_games = []
    with open('data/tourney_detailed_results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['season'])
            if season not in pseasons:
                wteam = int(row['wteam'])
                lteam = int(row['lteam'])
                wseed = seeds[wteam][season]
                lseed = seeds[lteam][season]
                rnd = slot(wseed, lseed)
                wrpi = rpi[wteam][season]
                lrpi = rpi[lteam][season]
                wrank = ranks[wteam][season]
                lrank = ranks[lteam][season]
                wavgstats = avg_stats(stats[wteam][season])
                lavgstats = avg_stats(stats[lteam][season])                
                if wteam < lteam:
                    wvector = [rnd, wrpi, lrpi, wrank, lrank] + wavgstats + lavgstats
                    games.append(wvector)
                    results.append(1)
                else:
                    lvector = [rnd, lrpi, wrpi, lrank, wrank] + lavgstats + wavgstats
                    games.append(lvector)
                    results.append(0)                       
    for matchup in pmatchups:
        season, teama, teamb = map(int, matchup.split('_'))
        aseed = seeds[teama][season]
        bseed = seeds[teamb][season]
        rnd = slot(aseed, bseed)
        arpi = rpi[teama][season]
        brpi = rpi[teamb][season]
        arank = ranks[teama][season]
        brank = ranks[teamb][season]
        aavgstats = avg_stats(stats[teama][season])
        bavgstats = avg_stats(stats[teamb][season])
        if teama < teamb:
            avector = [rnd, arpi, brpi, arank, brank] + aavgstats + bavgstats
            predict_games.append(avector)
        else:
            bvector = [rnd, brpi, arpi, brank, arank] + bavgstats + aavgstats
            predict_games.append(bvector)
    return numpy.array(games), numpy.array(results), numpy.array(predict_games)

def possible_matchups(pseasons, filename):
    matchups = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season, _, _ = map(int, row['id'].split('_'))
            if season in pseasons:
                matchups.append(row['id'])
    return matchups

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
def logloss(act, pred):
    return metrics.log_loss(act, pred)

def write_predictions(pmatchups, confidences):
    with open('submissions/submission.csv', 'wb') as csvfile:
        fieldnames = ['id', 'pred']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(pmatchups)):            
            writer.writerow({ 'id': pmatchups[i], 'pred': confidences[i] })
    teams = team_id_mapping()
    seeds = tourney_seeds()
    with open('submissions/bracket.txt', 'w') as f:
        for i in range(len(pmatchups)):
            season, teama, teamb = map(int, pmatchups[i].split('_'))
            #winner = numpy.random.choice([teama, teamb], 1, p=[confidences[i], 1.0-confidences[i]])[0]
            winner = teama if confidences[i] >= 0.5 else teamb           
            f.write(seeds[teama][season] + ' ' + teams[teama] + ' vs ' + seeds[teamb][season] + ' ' + 
                    teams[teamb] + ': ' + teams[winner] + ' ' + str(confidences[i]) + '\n')

def score_predictions(pmatchups, predictions):
    actual = []
    testable_predictions = []
    with open('data/tourney_detailed_results.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            season = int(row['season'])
            wteam = int(row['wteam'])
            lteam = int(row['lteam'])
            daynum = int(row['daynum'])
            if daynum > 135:
                if wteam < lteam:
                    matchup = str(season) + '_' + str(wteam) + '_' + str(lteam)
                    result = 1
                else:
                    matchup = str(season) + '_' + str(lteam) + '_' + str(wteam)
                    result = 0
                if matchup in pmatchups:
                    i = pmatchups.index(matchup)
                    predicted = predictions[i]
                    actual.append(result)
                    testable_predictions.append(predicted)
    print logloss(actual, testable_predictions)
    
# http://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#k-fold_cross-validation
# http://scikit-learn.org/stable/modules/grid_search.html
# http://scikit-learn.org/stable/modules/cross_validation.html#k-fold
def grid_search(X, y, classifier, param_grid):
    cv = StratifiedKFold(y=y, n_folds=5)
    grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    print grid.best_estimator_
    return grid.best_estimator_


predict_seasons = [2015]
predict_matchups = possible_matchups(predict_seasons, 'submissions/sample_submission.csv')

X, y, predict_X = data(predict_seasons, predict_matchups)

# http://en.wikipedia.org/wiki/Feature_scaling
# http://scikit-learn.org/stable/modules/preprocessing.html
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
predict_X = scaler.transform(predict_X)

# http://en.wikipedia.org/wiki/Feature_selection
# http://scikit-learn.org/stable/modules/feature_selection.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = grid_search(X, y, decomposition.PCA(), [{'n_components': range(5, X.shape[1])}])
X = pca.transform(X)
predict_X = pca.transform(predict_X)

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#rf = ExtraTreesClassifier(n_jobs=2, n_estimators=15000).fit(X, y)

# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn = grid_search(X, y, KNeighborsClassifier(), [{'n_neighbors': range(5, 25), 'metric': ['euclidean', 'manhattan']}])

# http://scikit-learn.org/stable/modules/svm.html
# linear uses stochastic gradient descent
# parameter C is related to regularization
#svm1 = grid_search(X, y, SVC(probability=True, kernel='rbf', tol=1e-4), [{'C': [0.1, 1, 10], 'gamma': [0.01, 0.001]} ])
svm2 = grid_search(X, y, SVC(probability=True, kernel='linear', tol=1e-4), [{'C': [0.1, 1, 10], 'gamma': [0.01, 0.001]} ])

# http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
log = grid_search(X, y, LogisticRegression(), [{'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}])

classifiers = [knn, log, svm2]
predicted_y_confidences = []

for classifier in classifiers:
    predictions = classifier.predict_proba(predict_X)[:,1]
    #score_predictions(predict_matchups, predictions)
    predicted_y_confidences.append(predictions)

# tried stacking with LogisticRegression, gave worse probabilities than median
avg_predicted_y_confidences = numpy.median(numpy.array(predicted_y_confidences), axis=0)

#score_predictions(predict_matchups, avg_predicted_y_confidences)
write_predictions(predict_matchups, avg_predicted_y_confidences)
