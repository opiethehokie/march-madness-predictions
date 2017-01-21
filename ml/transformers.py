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



from collections import defaultdict
from functools import partial

import numpy
import pandas

from scipy.stats import boxcox, skew
from sklearn.base import BaseEstimator, TransformerMixin

from ml import visualizations
from ml import wrangling


TOURNEY_START_DAY = 136

class HomeCourtTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.home = 'H'
        self.away = 'A'

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        onehot_home_court = []
        for row in X.itertuples(index=False):
            if row.Daynum >= TOURNEY_START_DAY:
                onehot_home_court.append([0, 0])
            elif row.Wloc != self.home and row.Wloc != self.away:
                onehot_home_court.append([0, 0])
            elif row.Wteam < row.Lteam:
                if row.Wloc == self.home:
                    onehot_home_court.append([1, 0])
                else:
                    onehot_home_court.append([0, 1])
            else:
                if row.Wloc == self.home:
                    onehot_home_court.append([0, 1])
                else:
                    onehot_home_court.append([1, 0])
        return numpy.array(onehot_home_court)

# http://netprophetblog.blogspot.com/2011/04/infinitely-deep-rpi.html
# http://netprophetblog.blogspot.com/2011/04/rpi-distribution.html
class ModifiedRPITransformer(BaseEstimator, TransformerMixin):

    def __init__(self, weights=(.15, .15, .7)):
        self.weights = weights
        self.stats = None

    def fit(self, X, _y=None):
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for row in X.itertuples(index=False):
            if row.Daynum < TOURNEY_START_DAY:
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
            win_percent = sum(results) / len(results)
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
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))

    def fit(self, X, _y=None):
        for row in X.itertuples(index=False):
            if row.Daynum < TOURNEY_START_DAY:
                self.stats[row.Season][row.Wteam]['points'].append(row.Wscore)
                self.stats[row.Season][row.Wteam]['points-against'].append(row.Lscore)
                self.stats[row.Season][row.Lteam]['points'].append(row.Lscore)
                self.stats[row.Season][row.Lteam]['points-against'].append(row.Wscore)
        return self

    def _pythagorean_expectation(self, results):
        points = numpy.sum(results['points'])
        points_against = numpy.sum(results['points-against'])
        return points**self.exponent / (points**self.exponent + points_against**self.exponent)

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

# in pipeline you can do something like: ('debug', DebugFeatureImportances(SelectFromModel(LinearSVC(penalty='l1', dual=False))))
class DebugFeatureImportances(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, _y=None):
        tmp = self.model.fit(X, _y)
        if hasattr(self.model, 'scores_'):
            print(sorted(self.model.scores_, reverse=True)[0: 50])
        elif hasattr(self.model.estimator_, 'coef'):
            print(sorted(numpy.abs(self.model.estimator_.coef_[0]), reverse=True)[0: 50])
        elif hasattr(self.model.estimator_, 'feature_importances_'):
            print(sorted(numpy.abs(self.model.estimator_.feature_importances_), reverse=True)[0: 50])
        return tmp

    def transform(self, X):
        return self.model.transform(X)

# in pipeline you can do something like: ('debug', DebugFeatureProperties())
# should also set n_jobs=1
class DebugFeatureProperties(BaseEstimator, TransformerMixin):

    def fit(self, X, _y=None):
        df = pandas.DataFrame(X)
        print(df.shape)
        print(df.describe())
        print(df.head(5))
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
        return X.iloc[:, self.start:self.end]

# http://netprophetblog.blogspot.com/2012/02/continued-slow-pursuit-of-statistical.html
class StatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stat_F):
        self.stat_F = stat_F
        self.stats = None

    #pylint: disable=too-many-statements
    def fit(self, X, _y=None):
        self.stats = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for row in X.itertuples(index=False):
            if row.Daynum < TOURNEY_START_DAY:
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
            if row.Wteam < row.Lteam:
                compiled_stats.append(numpy.hstack((wstats, lstats)))
            else:
                compiled_stats.append(numpy.hstack((lstats, wstats)))
        return numpy.array(compiled_stats)

class RatingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, rating_F, preseason_games, last_x_games=0):
        self.rating_F = rating_F
        self.preseason_games = preseason_games # need some games to have been played before ratings will converge
        self.last_x_games = last_x_games # exclude older games

        self.teams = None
        self.adj_stats = None

    def fit(self, X, _y=None):
        self.teams = wrangling.all_teams(pandas.concat([self.preseason_games, X], ignore_index=True))
        previous_games = defaultdict(partial(defaultdict, list))
        for row in self.preseason_games.itertuples(index=False):
            previous_games[row.Season][row.Wteam].append(row)
            previous_games[row.Season][row.Lteam].append(row)
        current_day = 0
        for row in X.itertuples(index=False):
            previous_games[row.Season][row.Wteam].append(row)
            previous_games[row.Season][row.Lteam].append(row)
            if row.Daynum < TOURNEY_START_DAY and row.Daynum > current_day:
                stats = self._recalc_stats(self.teams, previous_games, self.last_x_games)
                self.adj_stats = self.rating_F(stats)
                current_day = row.Daynum
        return self

    @staticmethod
    def _recalc_stats(teams, previous_games, last_x_games):
        stat_categories = {x:i for i, x in enumerate(['eff_field_goal_percent', 'rebound_rate', 'free_throw_rate', 'turnover_rate',
                                                      'assist_rate', 'block_rate', 'steal_rate', 'score_rate', 'foul_rate'])}
        stats = {}
        for season in list(teams.keys()):
            stats[season] = numpy.zeros((len(teams[season]), len(teams[season]), len(stat_categories)))
            for team in teams[season]:
                bss = previous_games[season][team][-last_x_games:]
                for bs in bss:
                    wteam_id = teams[season][bs.Wteam]
                    lteam_id = teams[season][bs.Lteam]
                    wposs = bs.Wfga - bs.Wor + bs.Wto + .475 * bs.Wfta
                    lposs = bs.Lfga - bs.Lor + bs.Lto + .475 * bs.Lfta
                    stats[season][wteam_id][lteam_id][stat_categories['eff_field_goal_percent']] += ((bs.Wfgm + .5 * bs.Wfgm3) / bs.Wfga)
                    stats[season][lteam_id][wteam_id][stat_categories['eff_field_goal_percent']] += ((bs.Lfgm + .5 * bs.Lfgm3) / bs.Lfga)
                    stats[season][wteam_id][lteam_id][stat_categories['rebound_rate']] += bs.Wor / (bs.Wor + bs.Ldr)
                    stats[season][lteam_id][wteam_id][stat_categories['rebound_rate']] += bs.Lor / (bs.Lor + bs.Wdr)
                    stats[season][wteam_id][lteam_id][stat_categories['free_throw_rate']] += (bs.Wfta / bs.Wftm) if bs.Wftm > 0 else 0
                    stats[season][lteam_id][wteam_id][stat_categories['free_throw_rate']] += (bs.Lfta / bs.Lftm) if bs.Lftm > 0 else 0
                    stats[season][wteam_id][lteam_id][stat_categories['turnover_rate']] += (bs.Wto / wposs)
                    stats[season][lteam_id][wteam_id][stat_categories['turnover_rate']] += (bs.Lto / lposs)
                    stats[season][wteam_id][lteam_id][stat_categories['assist_rate']] += (bs.Wast / bs.Wfgm)
                    stats[season][lteam_id][wteam_id][stat_categories['assist_rate']] += (bs.Last / bs.Lfgm)
                    stats[season][wteam_id][lteam_id][stat_categories['block_rate']] += (bs.Wblk / bs.Lfga)
                    stats[season][lteam_id][wteam_id][stat_categories['block_rate']] += (bs.Lblk / bs.Wfga)
                    stats[season][wteam_id][lteam_id][stat_categories['steal_rate']] += (bs.Wstl / lposs)
                    stats[season][lteam_id][wteam_id][stat_categories['steal_rate']] += (bs.Lstl / wposs)
                    stats[season][wteam_id][lteam_id][stat_categories['score_rate']] += (bs.Wscore / wposs)
                    stats[season][lteam_id][wteam_id][stat_categories['score_rate']] += (bs.Lscore / lposs)
                    stats[season][wteam_id][lteam_id][stat_categories['foul_rate']] += (bs.Wpf / lposs)
                    stats[season][lteam_id][wteam_id][stat_categories['foul_rate']] += (bs.Lpf / wposs)
        return stats

    def transform(self, X):
        adjusted = []
        for row in X.itertuples(index=False):
            team_ids = self.teams[row.Season]
            wstats = self.adj_stats[row.Season][team_ids[row.Wteam]]
            lstats = self.adj_stats[row.Season][team_ids[row.Lteam]]
            if row.Wteam < row.Lteam:
                adjusted.append(numpy.hstack((wstats, lstats)))
            else:
                adjusted.append(numpy.hstack((lstats, wstats)))
        return numpy.array(adjusted)

class SkewnessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, max_skew=.75, technique='log'):
        self.max_skew = max_skew
        self.technique = technique

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        X = pandas.DataFrame(X)
        skewed_feats = X.apply(skew)
        very_skewed_feats = skewed_feats[numpy.abs(skewed_feats) > self.max_skew]
        transformed = X.copy()
        if self.technique == 'sqrt':
            transformed[very_skewed_feats.index] = numpy.sqrt(X[very_skewed_feats.index])
        elif self.technique == 'log':
            transformed[very_skewed_feats.index] = numpy.log1p(X[very_skewed_feats.index])
        elif self.technique == 'boxcox':
            for idx in very_skewed_feats.index:
                transformed[idx] = boxcox(X[idx]+1)[0]
        return transformed

class OvertimeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, _X, _y=None):
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        for row in X.itertuples(index=True):
            if row.Numot > 0:
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
