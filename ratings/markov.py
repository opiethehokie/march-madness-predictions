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
