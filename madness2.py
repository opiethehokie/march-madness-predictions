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

import pandas as pd

from sklearn.metrics import log_loss

from db.kaggle import (read_raw_data, read_predictions, write_predictions, team_id_mapping, team_seed_mapping,
                       championship_pairings, possible_tourney_matchups)
from ml2.training import manual_regression_model
from ml2.preprocessing import (adjust_overtime_games, custom_train_test_split, filter_outlier_games, oversample_neutral_site_games,
                               tourney_mov_std, filter_out_of_window_games, add_features)
from ml2.postprocessing import mov_to_win_percent, override_final_predictions
from simulations.bracket import simulate_tourney


TOURNEY_DATA_FILE = 'data/tourney_detailed_results_2017.csv'
SEASON_DATA_FILE = 'data/regular_season_detailed_results_2018.csv'
SUBMISSION_FILE = 'results/submission.csv'
TEAMS_FILE = 'data/teams.csv'
SEEDS_FILE = 'data/seeds_2018.csv'
SLOTS_FILE = 'data/slots_2018.csv'


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2017

    SAMPLE_SUBMISSION_FILE = 'results/sample_submission_%s.csv' % predict_year
    TOURNEY_FORMAT_FILE = 'data/tourney_format_%s.yml' % predict_year

    start_year = predict_year - 5
    start_day = 30

    predict_matchups, future_games = possible_tourney_matchups(SAMPLE_SUBMISSION_FILE)

    games = read_raw_data(SEASON_DATA_FILE, TOURNEY_DATA_FILE)
    games = adjust_overtime_games(games)
    games = filter_outlier_games(games)
    games = oversample_neutral_site_games(games)
    games = filter_out_of_window_games(games, start_year, start_day, predict_year)
    games = pd.concat([games, future_games])

    data = add_features(games)

    X_train, X_test, X_predict, y_train, y_test = custom_train_test_split(data, predict_year)

    model = manual_regression_model(X_train, y_train)

    m = tourney_mov_std(games)
    if X_test.size > 0:
        results = model.predict(X_test)
        result_probas = [mov_to_win_percent(yi, m) for yi in results]
        print('Average log loss is %f' % log_loss(y_test, result_probas))

    predictions = model.predict(X_predict)
    prediction_probas = [mov_to_win_percent(yi, m) for yi in predictions]

    #write_predictions(predict_matchups, y_predict_probas, SUBMISSION_FILE)

    # post-processing for Kaggle competition (two submissions means we can always get championship game correct)
    #slots = championship_pairings(SLOTS_FILE)
    #seeds = team_seed_mapping(SEEDS_FILE)
    #write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, y_predict_probas, 0), '0')
    #write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, y_predict_probas, 1), '1')

    # predict actual tournament bracket for cash money
    #if predict_year >= 2015 and predict_year <= 2018:
    #    simulate_tourney(team_id_mapping(TEAMS_FILE), read_predictions(SUBMISSION_FILE), predict_year)
