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


import sys

import numpy as np

from sklearn.metrics import log_loss

from db.kaggle import (game_data, read_predictions, write_predictions, team_id_mapping, team_seed_mapping,
                       championship_pairings, possible_tourney_matchups)
from ml2.training import manual_regression_model, deep_learning_regression_model, auto_regression_model, average_predictions
from ml2.wrangling import (adjust_overtime_games, custom_train_test_split, filter_outlier_games, oversample_neutral_site_games,
                           filter_out_of_window_games, extract_features, add_games, fill_missing_stats, tourney_mov_std)
from ml2.postprocessing import override_final_predictions
from simulations.bracket import simulate_tourney


random_state = 42
np.random.seed(random_state)


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2017

    start_year = 2009
    start_day = 20

    predict_matchups, future_games = possible_tourney_matchups(predict_year)

    games = game_data()
    games = adjust_overtime_games(games)
    games = filter_outlier_games(games)
    games = oversample_neutral_site_games(games)
    games = add_games(games, future_games)
    games = fill_missing_stats(games)

    games_and_features = extract_features(games)
    games_and_features = filter_out_of_window_games(games_and_features, start_year, start_day, predict_year)

    X_train, X_test, X_predict, y_train, y_test = custom_train_test_split(games_and_features, predict_year)

    model1 = manual_regression_model(X_train, y_train, random_state)
    #model1 = deep_learning_regression_model(X_train, y_train, random_state)
    #model1 = auto_regression_model(X_train, y_train)

    if X_test.size > 0:
        result_probas = average_predictions([model1], X_test, tourney_mov_std(games))
        print('Test log loss: %f' % log_loss(y_test, result_probas))
    #prediction_probas = average_predictions([model1], X_predict, tourney_mov_std(games))
    #slots = championship_pairings()
    #seeds = team_seed_mapping()
    #write_predictions(predict_matchups, prediction_probas)
    #write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 0), '-0')
    #write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 1), '-1')
    #if predict_year >= 2015 and predict_year <= 2018:
    #    simulate_tourney(team_id_mapping(), read_predictions(), predict_year)
