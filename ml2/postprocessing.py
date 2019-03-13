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


import numpy as np

from scipy.stats import norm, normaltest, ttest_ind, ks_2samp


# two submissions means we can always get championship game correct
def override_final_predictions(slots, seeds, matchups, predictions, new_value):
    diff_predictions = list(predictions)
    for idx, matchup in enumerate(matchups):
        if _possible_tourney_final(slots, seeds, matchup):
            diff_predictions[idx] = new_value
    return diff_predictions

def _possible_tourney_final(slots, seeds, matchup):
    year, teama, teamb = matchup.split('_')
    teama_region = seeds[year][teama][0]
    teamb_region = seeds[year][teamb][0]
    (champ_regions1, champ_regions2) = slots[year]
    return ((champ_regions1.find(teama_region) > -1 and champ_regions2.find(teamb_region) > -1) or
            (champ_regions2.find(teama_region) > -1 and champ_regions1.find(teamb_region) > -1))

# https://www.pro-football-reference.com/about/win_prob.htm
def mov_to_win_percent(u, m=11, offset=0):
    u = u + offset
    return 1 - norm.cdf(0.5, loc=u, scale=m) + .5 * (norm.cdf(0.5, loc=u, scale=m) - norm.cdf(-0.5, loc=u, scale=m))

def average_predictions(models, X, mov_std):
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    results = np.mean(np.array(predictions), axis=0)
    return [mov_to_win_percent(yi, mov_std) for yi in results]

# https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
# https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/

# tells if one model is really better than another
def significance_test(vals1, vals2):
    if _normal(vals1) and _normal(vals2):
        return _t_test(vals1, vals2)
    return _ks_test(vals1, vals2)

def _normal(vals):
    _, p = normaltest(vals)
    return p >= 0.5

# null hypothesis H0 is that both samples drawn from same distribution, i.e. models work equally well
# false is same distribution (fail to reject H0 and no statistical significance), true is 95% confidence means are different
def _t_test(vals1, vals2):
    var = np.std(vals1) == np.std(vals2)
    _, p = ttest_ind(vals1, vals2, equal_var=var) # student is equal var, welch's is unequal var
    return p <= .05

def _ks_test(vals1, vals2):
    _, p = ks_2samp(vals1, vals2) # kolmogorov-smirnov
    return p <= .05

# tells how good final model is
def confidence_intervals(vals, alpha=.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(vals, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(vals, p))
    return lower, upper