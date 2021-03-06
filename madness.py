#   Copyright 2016-2020 Michael Peters
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


import random
import sys

import shap
import numpy as np

from sklearn.metrics import log_loss, roc_curve, confusion_matrix, auc, accuracy_score, classification_report

from db.kaggle import (game_data, read_predictions, write_predictions, team_id_mapping, team_seed_mapping,
                       championship_pairings, possible_tourney_matchups)
from ml.training import linear_model, boosting_model, neural_network_model, bayesian_model
from ml.wrangling import prepare_data, tourney_mov_std
from ml.postprocessing import (override_final_predictions, average_predictions, average_prediction_probas, significance_test,
                               confidence_intervals, effect_size, statistical_power, mov_to_win_percent)
from simulations.bracket import simulate_tourney


random_state = 4298
random.seed(random_state)
np.random.seed(random_state)


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2018

    start_year = 2009
    start_day = 60
    check_confidence = False
    save_predictions = False
    run_simulations = False
    explain_features = False

    games = game_data()
    m = tourney_mov_std(games)
    predict_matchups, future_games = possible_tourney_matchups(predict_year)

    X_train, X_test, X_predict, y_train, y_test, cv = prepare_data(games, future_games, start_day, start_year, predict_year)

    regression_y_train = [mov_to_win_percent(yi, m) for yi in y_train]
    regression_y_test = [mov_to_win_percent(yi, m) for yi in y_test]

    classification_y_train = [1 if yi > 0 else 0 for yi in y_train]
    classification_y_test = [1 if yi > 0 else 0 for yi in y_test]

    # ML

    regression_models = [neural_network_model(X_train, regression_y_train, cv, random_state, tune=False)
                        ]

    classification_models = [#linear_model(X_train, classification_y_train, cv, random_state, tune=False),
                             #boosting_model(X_train, classification_y_train, cv, random_state, tune=False),
                             #bayesian_model(X_train, classification_y_train, cv)
                            ]

    if X_test.size > 0:
        results = average_predictions(regression_models, classification_models, X_test)
        result_probas = average_prediction_probas(regression_models, classification_models, X_test)
        print('Test accuracy: %f' % accuracy_score(classification_y_test, results))
        print('Test confustion matrix:\n', confusion_matrix(classification_y_test, results))
        print('Test classification report:\n', classification_report(classification_y_test, results))
        fp_rates, tp_rates, _ = roc_curve(classification_y_test, results)
        print('Test AUC: %f' % auc(fp_rates, tp_rates))
        print('Test log loss: %f' % log_loss(classification_y_test, result_probas))

    if save_predictions:
        prediction_probas = average_prediction_probas(regression_models, classification_models, X_predict)
        slots = championship_pairings()
        seeds = team_seed_mapping()
        write_predictions(predict_matchups, prediction_probas)
        write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 0), '-0')
        write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 1), '-1')

    # data sci

    if check_confidence and X_test.size > 0:
        model1_results = []
        model2_results = []
        for predict_year in [2015, 2016, 2017, 2018]:
            _, future_games = possible_tourney_matchups(predict_year)
            X_train, X_test, X_predict, y_train, y_test, _ = prepare_data(games, future_games, start_day, start_year, predict_year)
            for rs in np.random.randint(0, 1000, 7):
                np.random.seed(rs)
                classification_y_train = [1 if yi > 0 else 0 for yi in y_train]
                classification_y_test = [1 if yi > 0 else 0 for yi in y_test]
                model1 = linear_model(X_train, classification_y_train, rs=rs, tune=False)
                model1_results.append(log_loss(classification_y_test, model1.predict_proba(X_test)[:, 1]))
                model2 = boosting_model(X_train, classification_y_train, rs=rs, tune=False)
                model2_results.append(log_loss(classification_y_test, model2.predict_proba(X_test)[:, 1]))
        # rough bootstrap with approx 30 samples
        print('Models are significantly different: ', significance_test(model1_results, model2_results))
        print('Effect size: ', effect_size(model1_results, model2_results))
        print('%d samples out of %f needed to see effect' % (len(model1_results), statistical_power()))
        print('95 percent confidence intervals for model 1: ', confidence_intervals(model1_results))
        print('95 percent confidence intervals for model 2: ', confidence_intervals(model2_results))
        lower, upper = confidence_intervals(np.mean(np.array([model1_results, model2_results]), axis=0))
        print('95 percent confidence intervals for average: %f - %f' % (lower, upper))

    if explain_features and X_test.size > 0:
        # game theoretic approach to global interpretability
        for model in regression_models:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 10))
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values)
        for model in classification_models:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 10))
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values)

    if run_simulations:
        simulate_tourney(team_id_mapping(), read_predictions(), predict_year)
