import numpy


def adjust_stats(stats):
    adjusted_stats = {}
    seasons = list(stats.keys())
    for season in seasons:
        cols = []
        for i in range(stats[season].shape[-1]):
            stat = stats[season][..., i]
            o, d = off_def_ratings(stat)
            cols.append(o)
            cols.append(d)
        adjusted_stats[season] = numpy.array(cols).transpose()
    return adjusted_stats

# http://netprophetblog.blogspot.com/2015/02/strength-of-schedule-adjusted_4.html
# http://biasvariance.blogspot.com/2015/07/blog-post.html
# http://meyer.math.ncsu.edu/Meyer/REU/REU2009/OffenseDefenseModel.pdf

def off_def_ratings(stat, perturbation=1e-4, threshold=.01):
    num_teams = len(stat)
    offensive_rating = offensive_rating_prev = numpy.zeros(num_teams)
    defensive_rating = defensive_rating_prev = numpy.ones(num_teams)
    stat += perturbation # small perturbation to help with convergence
    convergence = False
    iterations = 0
    max_iterations = 100 * (num_teams + 1)
    while not convergence and iterations < max_iterations:
        offensive_rating = (stat.transpose()/defensive_rating).sum(axis=1)
        defensive_rating = (stat/offensive_rating).sum(axis=1)
        convergence = (numpy.allclose(offensive_rating, offensive_rating_prev, threshold) and
                       numpy.allclose(defensive_rating, defensive_rating_prev, threshold))
        offensive_rating_prev = offensive_rating
        defensive_rating_prev = defensive_rating
        iterations += 1
    if iterations == max_iterations:
        raise Exception('no convergence on offensive-defensive ratings')
    return offensive_rating, defensive_rating
