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


import os

import numpy
import pandas

from ratings import markov


numpy.seterr(all='raise')

def soccer_goals():
    teams = ['Aston Villa', 'Arsenal', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Hull',
             'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Southampton', 'Stoke',
             'Sunderland', 'Swansea', 'Spurs', 'West Ham', 'West Brom', 'QPR']
    scores = numpy.zeros((len(teams), len(teams)))
    goals = pandas.DataFrame(scores, index=teams, columns=teams)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data = pandas.read_csv(this_dir + '/premier_league.csv')
    for _, datam in data.iterrows():
        if datam.home_team in teams and datam.away_team in teams:
            goals[datam.home_team][datam.away_team] += float(datam.home_goals)
            goals[datam.away_team][datam.home_team] += float(datam.away_goals)
    return goals

def test_markov_ratings():
    actual = markov.markov_ratings(soccer_goals())
    expected = [0.03313267, 0.0666793, 0.03089082, 0.07173901, 0.0486637,
                0.04862142, 0.03690847, 0.04821738, 0.05771672, 0.07952477,
                0.06285796, 0.04087315, 0.04876876, 0.05212288, 0.03543908,
                0.05145822, 0.06033652, 0.04430509, 0.04025114, 0.04149294]
    assert numpy.allclose(expected, actual, .0001)
