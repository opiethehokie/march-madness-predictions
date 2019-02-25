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


from scipy.stats import norm


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

def significance_test():
    pass

def confidence_intervals():
    pass
