#   Copyright 2016-2018 Michael Peters
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
from functools import partial

import numpy
import pandas

from scipy.stats import boxcox, skew
from sklearn.base import BaseEstimator, TransformerMixin

from ml import wrangling
from ml import visualizations


TOURNEY_START_DAY = 136

class HomeCourtTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, factor=.96):
        self.factor = factor

    def fit(self, _X, _y=None):
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        X = X.copy()
        for row in X.itertuples(index=True):
            if row.Daynum < TOURNEY_START_DAY and row.Wloc != 'N':
                wadj = self.factor if row.Wloc == 'H' else 1 / self.factor
                ladj = 2 - self.factor if row.Wloc == 'H' else self.factor
                X.set_value(row.Index, 'Wscore', row.Wscore * wadj)
                X.set_value(row.Index, 'Lscore', row.Lscore * ladj)
                X.set_value(row.Index, 'Wfgm', row.Wfgm * wadj)
                X.set_value(row.Index, 'Lfgm', row.Lfgm * ladj)
                X.set_value(row.Index, 'Wfga', row.Wfga * wadj)
                X.set_value(row.Index, 'Lfga', row.Lfga * ladj)
                X.set_value(row.Index, 'Wfgm3', row.Wfgm3 * wadj)
                X.set_value(row.Index, 'Lfgm3', row.Lfgm3 * ladj)
                X.set_value(row.Index, 'Wfga3', row.Wfga3 * wadj)
                X.set_value(row.Index, 'Lfga3', row.Lfga3 * ladj)
                X.set_value(row.Index, 'Wftm', row.Wftm * wadj)
                X.set_value(row.Index, 'Lftm', row.Lftm * ladj)
                X.set_value(row.Index, 'Wfta', row.Wfta * wadj)
                X.set_value(row.Index, 'Lfta', row.Lfta * ladj)
                X.set_value(row.Index, 'Wor', row.Wor * wadj)
                X.set_value(row.Index, 'Lor', row.Lor * ladj)
                X.set_value(row.Index, 'Wdr', row.Wdr * wadj)
                X.set_value(row.Index, 'Ldr', row.Ldr * ladj)
                X.set_value(row.Index, 'Wast', row.Wast * wadj)
                X.set_value(row.Index, 'Last', row.Last * ladj)
                X.set_value(row.Index, 'Wto', row.Wto * wadj)
                X.set_value(row.Index, 'Lto', row.Lto * ladj)
                X.set_value(row.Index, 'Wstl', row.Wstl * wadj)
                X.set_value(row.Index, 'Lstl', row.Lstl * ladj)
                X.set_value(row.Index, 'Wblk', row.Wblk * wadj)
                X.set_value(row.Index, 'Lblk', row.Lblk * ladj)
                X.set_value(row.Index, 'Wpf', row.Wpf * wadj)
                X.set_value(row.Index, 'Lpf', row.Lpf * ladj)
        return X

# http://netprophetblog.blogspot.com/2011/04/infinitely-deep-rpi.html
# http://netprophetblog.blogspot.com/2011/04/rpi-distribution.html
class ModifiedRPITransformer(BaseEstimator, TransformerMixin):

    def __init__(self, weights=(.15, .15, .7)):
        self.weights = weights
        self.stats = None

    def fit(self, X, _y=None):
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for row in X.itertuples(index=False):
            self.stats[row.Season][row.Wteam]['opponents'].append(row.Lteam)
            self.stats[row.Season][row.Lteam]['opponents'].append(row.Wteam)
            self.stats[row.Season][row.Wteam]['results'].append(1)
            self.stats[row.Season][row.Lteam]['results'].append(0)
        return self

    def _rpi(self, season_results, team):
        results = season_results[team]['results']
        win_percent = sum(results) / len(results) if len(results) > 0 else 0
        opponents_win_percent = self._opponents_win_percent(season_results, [o for o in season_results[team]['opponents'] if o != team])
        opponents_opponents_win_percent = self._opponents_opponents_win_percent(season_results, season_results[team]['opponents'])
        w1, w2, w3 = self.weights[0], self.weights[1], self.weights[2]
        return w1 * win_percent + w2 * opponents_win_percent + w3 * opponents_opponents_win_percent

    @staticmethod
    def _opponents_win_percent(season_results, opponents):
        win_percents = []
        for opponent in opponents:
            results = season_results[opponent]['results']
            win_percent = sum(results) / len(results) if len(results) > 0 else 0
            win_percents.append(win_percent)
        return sum(win_percents) / len(opponents) if len(opponents) > 0 else 0

    def _opponents_opponents_win_percent(self, season_results, opponents):
        win_percents = []
        for opponent in opponents:
            win_percent = self._opponents_win_percent(season_results, season_results[opponent]['opponents'])
            win_percents.append(win_percent)
        return sum(win_percents) / len(opponents) if len(opponents) > 0 else 0

    def transform(self, X):
        rpis = []
        for row in X.itertuples(index=False):
            wrpi = self._rpi(self.stats[row.Season], row.Wteam)
            lrpi = self._rpi(self.stats[row.Season], row.Lteam)
            if row.Wteam < row.Lteam:
                rpis.append([wrpi, lrpi])
            else:
                rpis.append([lrpi, wrpi])
        return numpy.array(rpis)

class PythagoreanExpectationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, exponent=10.25):
        self.exponent = exponent
        self.stats = None

    def fit(self, X, _y=None):
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for row in X.itertuples(index=False):
            self.stats[row.Season][row.Wteam]['points'].append(row.Wscore)
            self.stats[row.Season][row.Wteam]['points-against'].append(row.Lscore)
            self.stats[row.Season][row.Lteam]['points'].append(row.Lscore)
            self.stats[row.Season][row.Lteam]['points-against'].append(row.Wscore)
        return self

    def _pythagorean_expectation(self, results):
        points = numpy.sum(results['points'])
        points_against = numpy.sum(results['points-against'])
        numerator = points**self.exponent
        denomenator = points**self.exponent + points_against**self.exponent
        return numerator / denomenator if denomenator != 0 else 0

    def transform(self, X):
        pythags = []
        for row in X.itertuples(index=False):
            wpythag = self._pythagorean_expectation(self.stats[row.Season][row.Wteam])
            lpythag = self._pythagorean_expectation(self.stats[row.Season][row.Lteam])
            if row.Wteam < row.Lteam:
                pythags.append([wpythag, lpythag])
            else:
                pythags.append([lpythag, wpythag])
        return numpy.array(pythags)

# in pipeline you can do something like: ('debug', DebugFeatureProperties())
# should also set n_jobs=1
class DebugFeatureProperties(BaseEstimator, TransformerMixin):

    def fit(self, X, _y=None):
        df = pandas.DataFrame(X)
        print(df.shape)
        print(df.head(3))
        print(df.describe())
        print(skew(df))
        print(df.cov())
        print(df.corr())
        visualizations.plot_scatter_matrix(df)
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        return X

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        df = pandas.DataFrame(X)
        return df.iloc[:, self.start:self.end]

# http://netprophetblog.blogspot.com/2012/02/continued-slow-pursuit-of-statistical.html
class StatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stat_F):
        self.stat_F = stat_F
        self.stats = None

    #pylint: disable=too-many-statements
    def fit(self, X, _y=None):
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for row in X.itertuples(index=False):
            self.stats[row.Season][row.Wteam]['score'].append(row.Wscore)
            self.stats[row.Season][row.Wteam]['score-against'].append(row.Lscore)
            self.stats[row.Season][row.Lteam]['score'].append(row.Lscore)
            self.stats[row.Season][row.Lteam]['score-against'].append(row.Wscore)
            self.stats[row.Season][row.Wteam]['fgm'].append(row.Wfgm)
            self.stats[row.Season][row.Wteam]['fgm-against'].append(row.Lfgm)
            self.stats[row.Season][row.Lteam]['fgm'].append(row.Lfgm)
            self.stats[row.Season][row.Lteam]['fgm-against'].append(row.Wfgm)
            self.stats[row.Season][row.Wteam]['fga'].append(row.Wfga)
            self.stats[row.Season][row.Wteam]['fga-against'].append(row.Lfga)
            self.stats[row.Season][row.Lteam]['fga'].append(row.Lfga)
            self.stats[row.Season][row.Lteam]['fga-against'].append(row.Wfga)
            self.stats[row.Season][row.Wteam]['fgm3'].append(row.Wfgm3)
            self.stats[row.Season][row.Wteam]['fgm3-against'].append(row.Lfgm3)
            self.stats[row.Season][row.Lteam]['fgm3'].append(row.Lfgm3)
            self.stats[row.Season][row.Lteam]['fgm3-against'].append(row.Wfgm3)
            self.stats[row.Season][row.Wteam]['fga3'].append(row.Wfga3)
            self.stats[row.Season][row.Wteam]['fga3-against'].append(row.Lfga3)
            self.stats[row.Season][row.Lteam]['fga3'].append(row.Lfga3)
            self.stats[row.Season][row.Lteam]['fga3-against'].append(row.Wfga3)
            self.stats[row.Season][row.Wteam]['ftm'].append(row.Wftm)
            self.stats[row.Season][row.Wteam]['ftm-against'].append(row.Lftm)
            self.stats[row.Season][row.Lteam]['ftm'].append(row.Lftm)
            self.stats[row.Season][row.Lteam]['ftm-against'].append(row.Wftm)
            self.stats[row.Season][row.Wteam]['fta'].append(row.Wfta)
            self.stats[row.Season][row.Wteam]['fta-against'].append(row.Lfta)
            self.stats[row.Season][row.Lteam]['fta'].append(row.Lfta)
            self.stats[row.Season][row.Lteam]['fta-against'].append(row.Wfta)
            self.stats[row.Season][row.Wteam]['or'].append(row.Wor)
            self.stats[row.Season][row.Wteam]['or-against'].append(row.Lor)
            self.stats[row.Season][row.Lteam]['or'].append(row.Lor)
            self.stats[row.Season][row.Lteam]['or-against'].append(row.Wor)
            self.stats[row.Season][row.Wteam]['dr'].append(row.Wdr)
            self.stats[row.Season][row.Wteam]['dr-against'].append(row.Ldr)
            self.stats[row.Season][row.Lteam]['dr'].append(row.Ldr)
            self.stats[row.Season][row.Lteam]['dr-against'].append(row.Wdr)
            self.stats[row.Season][row.Wteam]['ast'].append(row.Wast)
            self.stats[row.Season][row.Wteam]['ast-against'].append(row.Last)
            self.stats[row.Season][row.Lteam]['ast'].append(row.Last)
            self.stats[row.Season][row.Lteam]['ast-against'].append(row.Wast)
            self.stats[row.Season][row.Wteam]['to'].append(row.Wto)
            self.stats[row.Season][row.Wteam]['to-against'].append(row.Lto)
            self.stats[row.Season][row.Lteam]['to'].append(row.Lto)
            self.stats[row.Season][row.Lteam]['to-against'].append(row.Wto)
            self.stats[row.Season][row.Wteam]['stl'].append(row.Wstl)
            self.stats[row.Season][row.Wteam]['stl-against'].append(row.Lstl)
            self.stats[row.Season][row.Lteam]['stl'].append(row.Lstl)
            self.stats[row.Season][row.Lteam]['stl-against'].append(row.Wstl)
            self.stats[row.Season][row.Wteam]['blk'].append(row.Wblk)
            self.stats[row.Season][row.Wteam]['blk-against'].append(row.Lblk)
            self.stats[row.Season][row.Lteam]['blk'].append(row.Lblk)
            self.stats[row.Season][row.Lteam]['blk-against'].append(row.Wblk)
            self.stats[row.Season][row.Wteam]['pf'].append(row.Wpf)
            self.stats[row.Season][row.Wteam]['pf-against'].append(row.Lpf)
            self.stats[row.Season][row.Lteam]['pf'].append(row.Lpf)
            self.stats[row.Season][row.Lteam]['pf-against'].append(row.Wpf)
        return self

    def transform(self, X):
        compiled_stats = []
        for row in X.itertuples(index=False):
            wstats = self.stat_F(self.stats[row.Season][row.Wteam])
            lstats = self.stat_F(self.stats[row.Season][row.Lteam])
            if len(wstats) == 0:
                wstats = [0] * int(len(compiled_stats[0]) / 2)
            if len(lstats) == 0:
                lstats = [0] * int(len(compiled_stats[0]) / 2)
            if row.Wteam < row.Lteam:
                compiled_stats.append(numpy.hstack((wstats, lstats)))
            else:
                compiled_stats.append(numpy.hstack((lstats, wstats)))
        return numpy.array(compiled_stats)

class RatingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, rating_F, preseason_games=None):
        self.rating_F = rating_F
        self.preseason_games = preseason_games
        self.adj_stats = None
        self.teams = None

    def fit(self, X, _y=None):
        if self.preseason_games is not None:
            X = pandas.concat([self.preseason_games, X], ignore_index=True)
        self.teams = wrangling.all_teams(X)
        seasons = numpy.unique(X['Season'])
        stat_categories = {x:i for i, x in enumerate(['eff_field_goal_percent', 'true_shooting', 'rebound_rate', 'free_throw_rate',
                                                      'turnover_rate', 'assist_rate', 'block_rate', 'steal_rate', 'score_rate', 'foul_rate'])}
        stats = {season: numpy.zeros((len(self.teams[season]), len(self.teams[season]), len(stat_categories))) for season in seasons}
        for row in X.itertuples(index=False):
            wteam_id = self.teams[row.Season][row.Wteam]
            lteam_id = self.teams[row.Season][row.Lteam]
            wposs = row.Wfga - row.Wor + row.Wto + .475 * row.Wfta
            lposs = row.Lfga - row.Lor + row.Lto + .475 * row.Lfta
            stats[row.Season][wteam_id][lteam_id][stat_categories['eff_field_goal_percent']] += ((row.Wfgm + .5 * row.Wfgm3) / row.Wfga)
            stats[row.Season][lteam_id][wteam_id][stat_categories['eff_field_goal_percent']] += ((row.Lfgm + .5 * row.Lfgm3) / row.Lfga)
            stats[row.Season][wteam_id][lteam_id][stat_categories['true_shooting']] += (.5 * row.Wscore / (row.Wfga + 0.475 * row.Wfta))
            stats[row.Season][lteam_id][wteam_id][stat_categories['true_shooting']] += (.5 * row.Lscore / (row.Lfga + 0.475 * row.Lfta))
            stats[row.Season][wteam_id][lteam_id][stat_categories['rebound_rate']] += row.Wor / (row.Wor + row.Ldr)
            stats[row.Season][lteam_id][wteam_id][stat_categories['rebound_rate']] += row.Lor / (row.Lor + row.Wdr)
            stats[row.Season][wteam_id][lteam_id][stat_categories['free_throw_rate']] += (row.Wfta / row.Wftm) if row.Wftm > 0 else 0
            stats[row.Season][lteam_id][wteam_id][stat_categories['free_throw_rate']] += (row.Lfta / row.Lftm) if row.Lftm > 0 else 0
            stats[row.Season][wteam_id][lteam_id][stat_categories['turnover_rate']] += (row.Wto / wposs)
            stats[row.Season][lteam_id][wteam_id][stat_categories['turnover_rate']] += (row.Lto / lposs)
            stats[row.Season][wteam_id][lteam_id][stat_categories['assist_rate']] += (row.Wast / row.Wfgm)
            stats[row.Season][lteam_id][wteam_id][stat_categories['assist_rate']] += (row.Last / row.Lfgm)
            stats[row.Season][wteam_id][lteam_id][stat_categories['block_rate']] += (row.Wblk / row.Lfga)
            stats[row.Season][lteam_id][wteam_id][stat_categories['block_rate']] += (row.Lblk / row.Wfga)
            stats[row.Season][wteam_id][lteam_id][stat_categories['steal_rate']] += (row.Wstl / lposs)
            stats[row.Season][lteam_id][wteam_id][stat_categories['steal_rate']] += (row.Lstl / wposs)
            stats[row.Season][wteam_id][lteam_id][stat_categories['score_rate']] += (row.Wscore / wposs)
            stats[row.Season][lteam_id][wteam_id][stat_categories['score_rate']] += (row.Lscore / lposs)
            stats[row.Season][wteam_id][lteam_id][stat_categories['foul_rate']] += (row.Wpf / lposs)
            stats[row.Season][lteam_id][wteam_id][stat_categories['foul_rate']] += (row.Lpf / wposs)
        self.adj_stats = self.rating_F(stats)
        return self

    def transform(self, X):
        ratings = []
        for row in X.itertuples(index=False):
            team_ids = self.teams[row.Season]
            wrating = self.adj_stats[row.Season][team_ids[row.Wteam]]
            lrating = self.adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                ratings.append(numpy.hstack((wrating, lrating)))
            else:
                ratings.append(numpy.hstack((lrating, wrating)))
        return numpy.array(ratings)

class SimpleRatingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, rating_F, preseason_games=None):
        self.rating_F = rating_F
        self.preseason_games = preseason_games
        self.adj_stats = None
        self.teams = None

    def fit(self, X, _y=None):
        if self.preseason_games is not None:
            X = pandas.concat([self.preseason_games, X], ignore_index=True)
        self.teams = wrangling.all_teams(X)
        seasons = numpy.unique(X['Season'])
        stat_categories = {x:i for i, x in enumerate(['points'])}
        stats = {season: numpy.zeros((len(self.teams[season]), len(self.teams[season]), len(stat_categories))) for season in seasons}
        for row in X.itertuples(index=False):
            wteam_id = self.teams[row.Season][row.Wteam]
            lteam_id = self.teams[row.Season][row.Lteam]
            stats[row.Season][wteam_id][lteam_id][stat_categories['points']] += row.Wscore
            stats[row.Season][lteam_id][wteam_id][stat_categories['points']] += row.Lscore
        self.adj_stats = self.rating_F(stats)
        return self

    def transform(self, X):
        ratings = []
        for row in X.itertuples(index=False):
            team_ids = self.teams[row.Season]
            wrating = self.adj_stats[row.Season][team_ids[row.Wteam]]
            lrating = self.adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                ratings.append(numpy.hstack((wrating, lrating)))
            else:
                ratings.append(numpy.hstack((lrating, wrating)))
        return numpy.array(ratings)

class SkewnessTransformer(BaseEstimator, TransformerMixin):

    # lmbda 0 is log
    # lmbda .5 is square root
    # lmbda 1 is no transform
    # lmbda None is statistically tuned
    def __init__(self, max_skew=2.5, lmbda=None):
        self.max_skew = max_skew
        self.lmbda = lmbda

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        X = pandas.DataFrame(X)
        skewed_feats = X.apply(skew)
        very_skewed_feats = skewed_feats[numpy.abs(skewed_feats) > self.max_skew]
        transformed = X.copy()
        for idx in very_skewed_feats.index:
            transformed[idx] = boxcox(X[idx]+1, lmbda=self.lmbda)[0]
        return transformed

class OvertimeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, _X, _y=None):
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        X = X.copy()
        for row in X.itertuples(index=True):
            if row.Daynum < TOURNEY_START_DAY and row.Numot > 0:
                ot_adj = 40 / (40 + row.Numot * 5)
                X.set_value(row.Index, 'Wscore', row.Wscore * ot_adj)
                X.set_value(row.Index, 'Lscore', row.Lscore * ot_adj)
                X.set_value(row.Index, 'Wfgm', row.Wfgm * ot_adj)
                X.set_value(row.Index, 'Lfgm', row.Lfgm * ot_adj)
                X.set_value(row.Index, 'Wfga', row.Wfga * ot_adj)
                X.set_value(row.Index, 'Lfga', row.Lfga * ot_adj)
                X.set_value(row.Index, 'Wfgm3', row.Wfgm3 * ot_adj)
                X.set_value(row.Index, 'Lfgm3', row.Lfgm3 * ot_adj)
                X.set_value(row.Index, 'Wfga3', row.Wfga3 * ot_adj)
                X.set_value(row.Index, 'Lfga3', row.Lfga3 * ot_adj)
                X.set_value(row.Index, 'Wftm', row.Wftm * ot_adj)
                X.set_value(row.Index, 'Lftm', row.Lftm * ot_adj)
                X.set_value(row.Index, 'Wfta', row.Wfta * ot_adj)
                X.set_value(row.Index, 'Lfta', row.Lfta * ot_adj)
                X.set_value(row.Index, 'Wor', row.Wor * ot_adj)
                X.set_value(row.Index, 'Lor', row.Lor * ot_adj)
                X.set_value(row.Index, 'Wdr', row.Wdr * ot_adj)
                X.set_value(row.Index, 'Ldr', row.Ldr * ot_adj)
                X.set_value(row.Index, 'Wast', row.Wast * ot_adj)
                X.set_value(row.Index, 'Last', row.Last * ot_adj)
                X.set_value(row.Index, 'Wto', row.Wto * ot_adj)
                X.set_value(row.Index, 'Lto', row.Lto * ot_adj)
                X.set_value(row.Index, 'Wstl', row.Wstl * ot_adj)
                X.set_value(row.Index, 'Lstl', row.Lstl * ot_adj)
                X.set_value(row.Index, 'Wblk', row.Wblk * ot_adj)
                X.set_value(row.Index, 'Lblk', row.Lblk * ot_adj)
                X.set_value(row.Index, 'Wpf', row.Wpf * ot_adj)
                X.set_value(row.Index, 'Lpf', row.Lpf * ot_adj)
        return X
