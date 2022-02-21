import numpy
import pandas

from ratings import off_def
#pylint: disable=unused-import
from ratings.tests.test_markov import soccer_goals


def test_off_def_ratings():
    teams = ['blue', 'gold', 'silver']
    num_teams = len(teams)
    stat = numpy.zeros((num_teams, num_teams))
    df = pandas.DataFrame(stat, index=teams, columns=teams)
    df['gold']['silver'] = .43
    df['silver']['gold'] = .3
    df['gold']['blue'] = .35
    df['blue']['gold'] = .23
    df['silver']['blue'] = .28
    df['blue']['silver'] = .26
    offensive, defensive = off_def.off_def_ratings(df)
    assert numpy.allclose(offensive, [0.47, 0.77, 0.61], .01)
    assert numpy.allclose(defensive, [0.91, 0.98, 1.11], .01)
    offensive, defensive = off_def.off_def_ratings(df.values)
    assert numpy.allclose(offensive, [0.47, 0.77, 0.61], .01)
    assert numpy.allclose(defensive, [0.91, 0.98, 1.11], .01)

#pylint: disable=redefined-outer-name
def test_off_def_ratings2(soccer_goals):
    actual_o, actual_d = off_def.off_def_ratings(soccer_goals.values)
    expected_o = numpy.array([31.33843405, 68.37509144, 27.76305066, 72.38060068, 47.09823379,
                              47.49119705, 33.00865301, 46.17519618, 53.39592662, 82.09304749,
                              60.90819241, 40.48603638, 51.71478376, 48.63047941, 31.81090453,
                              46.85475423, 58.97830649, 43.7611087, 38.50385291, 41.91005827])
    expected_d = numpy.array([1.11070601, 0.73745433, 1.10473189, 0.71818717, 1.1241924,
                              1.05787969, 1.03376991, 1.09573768, 0.92446061, 0.87012878,
                              0.79235822, 1.2685864, 0.66998681, 0.95158258, 1.07358027,
                              0.9183903, 1.0844594, 0.94361763, 1.05815724, 1.46842408])
    assert numpy.allclose(expected_o, actual_o, .00005)
    assert numpy.allclose(expected_d, actual_d, .00005)

def test_adjust_stats():
    stats = {2010: numpy.array([[[0, .23, .3],
                                 [.35, 0, .28],
                                 [.43, .26, 0]]]).reshape((3, 3, 1))}
    adj_stats = off_def.adjust_stats(stats)
    assert numpy.allclose(adj_stats[2010], [[0.771, 0.980],
                                            [0.468, 0.911],
                                            [0.613, 1.112]], .01)
