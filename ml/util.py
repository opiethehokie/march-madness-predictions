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


import cProfile

from scipy.stats import norm


# annotate a function with @profile to see where it's spending the most time
def profile(func):
    def profiled_func(*args, **kwargs):
        p = cProfile.Profile()
        try:
            p.enable()
            result = func(*args, **kwargs)
            p.disable()
            return result
        finally:
            p.print_stats()
    return profiled_func

# annotate a function with @print_models
def print_models(func):
    def printed_func(*args, **kwargs):
        model = func(*args, **kwargs)
        cv_keys = ('mean_test_score', 'std_test_score', 'params')
        for r, _ in enumerate(model.cv_results_['mean_test_score']):
            print("%0.3f +/- %0.2f %r" % (model.cv_results_[cv_keys[0]][r],
                                          model.cv_results_[cv_keys[1]][r] / 2.0,
                                          model.cv_results_[cv_keys[2]][r]))
        print('Best parameters: %s' % model.best_params_)
        print('Best accuracy: %.2f' % model.best_score_)
        return model
    return printed_func

# https://www.pro-football-reference.com/about/win_prob.htm
def mov_to_win_percent(u, m=11, offset=0):
    u = u + offset
    return 1 - norm.cdf(0.5, loc=u, scale=m) + .5 * (norm.cdf(0.5, loc=u, scale=m) - norm.cdf(-0.5, loc=u, scale=m))
