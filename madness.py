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

from sklearn.metrics import log_loss

from db.kaggle import game_data, possible_tourney_matchups
from ml2.training import manual_regression_model, auto_regression_model
from ml2.wrangling import (adjust_overtime_games, custom_train_test_split, filter_outlier_games, oversample_neutral_site_games,
                           filter_out_of_window_games, extract_features, add_games, fill_missing_stats, tourney_mov_std)
from ml2.postprocessing import significance_test, confidence_intervals, mov_to_win_percent


if __name__ == '__main__':

    predict_years = [2018]

    start_year = 2009
    start_day = 20

    model1_results = []
    model2_results = []

    for predict_year in predict_years:

        predict_matchups, future_games = possible_tourney_matchups(predict_year)

        games = game_data()
        games = adjust_overtime_games(games)
        games = filter_outlier_games(games)
        games = oversample_neutral_site_games(games)
        games = add_games(games, future_games)
        games = fill_missing_stats(games)

        features = extract_features(games)
        features = filter_out_of_window_games(features, start_year, start_day, predict_year)
        X_train, X_test, _, y_train, y_test = custom_train_test_split(features, predict_year)

        mov_std = tourney_mov_std(games)

        for rs in np.random.randint(0, 1000, 10):
            model1 = manual_regression_model(X_train, y_train, random_state=int(rs), tune=False)
            model1_results.append(log_loss(y_test, [mov_to_win_percent(yi, mov_std) for yi in model1.predict(X_test)]))
            model2 = auto_regression_model(X_train, y_train, random_state=int(rs), tune=False)
            model2_results.append(log_loss(y_test, [mov_to_win_percent(yi, mov_std) for yi in model2.predict(X_test)]))

    print('Year %i' % predict_year)
    print('Models are significantly different: ', significance_test(model1_results, model2_results))
    print('95 percent confidence intervals for manual regression: ', confidence_intervals(model1_results))
    print('95 percent confidence intervals for auto regression: ', confidence_intervals(model2_results))
    lower, upper = confidence_intervals(np.mean(np.array([model1_results, model2_results]), axis=0))
    print('95 percent confidence intervals for average: %f - %f' % (lower, upper))
    print()
