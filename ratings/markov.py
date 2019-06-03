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


import numpy


# http://biasvariance.blogspot.com/2015/07/premier-league-2014-15-season-rankings.html
# https://github.com/jmmcd/GPDistance/blob/master/python/RandomWalks/ergodic.py

def markov_stats(stats):
    ratings = {}
    seasons = list(stats.keys())
    for season in seasons:
        cols = []
        for i in range(stats[season].shape[-1]):
            stat = stats[season][..., i]
            cols.append(markov_ratings(stat))
        ratings[season] = numpy.array(cols).T
    return ratings

def markov_ratings(stat, perturbation=1e-4):
    stat += perturbation # small perturbation to help with convergence
    transition_matrix = (stat.T / stat.sum(axis=1)).T
    w, v = numpy.linalg.eig(transition_matrix.T)
    i = w.argmax()
    return (v[:, i] / sum(v[:, i])).real
