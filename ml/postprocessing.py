from math import sqrt

import numpy as np

from scipy.stats import norm, normaltest, ttest_ind, ks_2samp
from statsmodels.stats.power import TTestIndPower


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

#TODO could clip less for R1 (.99) and more for R2 (.93) based on historical upset probabilities
def average_prediction_probas(regression_models, classification_models, X, low_clip=.025, high_clip=.975):
    predictions = [model.predict(X) for model in regression_models] + [model.predict_proba(X)[:, -1] for model in classification_models]
    return np.clip(np.mean(np.array(predictions), axis=0), low_clip, high_clip)

def average_predictions(regression_models, classification_models, X):
    predictions = [np.reshape([1 if yi >= .5 else 0 for yi in model.predict(X)], (X.shape[0])) for model in regression_models] + \
                  [np.reshape(model.predict(X), (X.shape[0])) for model in classification_models]
    return np.rint(np.mean(np.array(predictions), axis=0))

# https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
# https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/

# tells if one model is really better than another
# null hypothesis H0 is that both samples drawn from same distribution, i.e. models work equally well
# false is same distribution (fail to reject H0 and no statistical significance), true is 95% confidence means are different
# type I error is erroneously rejecting the null hypothesis
# type II error is erroneously not rejecting the null hypothesis
def significance_test(vals1, vals2):
    if _normal(vals1) and _normal(vals2):
        return _t_test(vals1, vals2) <= .05
    return _ks_test(vals1, vals2) <= .05

def _normal(vals):
    _, p = normaltest(vals)
    return p >= 0.5

def _t_test(vals1, vals2):
    var = np.std(vals1) == np.std(vals2)
    _, p = ttest_ind(vals1, vals2, equal_var=var) # student is equal var, welch's is unequal var
    return p

def _ks_test(vals1, vals2):
    _, p = ks_2samp(vals1, vals2) # kolmogorov-smirnov
    return p

# tells how good final model is
def confidence_intervals(vals, alpha=.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(vals, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(vals, p))
    return lower, upper

# https://www.kdnuggets.com/2019/01/comparing-machine-learning-models-statistical-vs-practical-significance.html

# quantifies size of effect if statistically significant, small if less than .1 or .3
def effect_size(vals1, vals2):
    p = _t_test(vals1, vals2) if _normal(vals1) and _normal(vals2) else _ks_test(vals1, vals2)
    return abs(norm.ppf(p)) / sqrt(len(vals1))

# probability of true positive (only useful when null hypothesis rejected), used to estimate min sample size
# defaults 20% chance of type II error, 5% chance type I error
def statistical_power(effect=.8, power=.8, alpha=.05):
    analysis = TTestIndPower()
    return analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
