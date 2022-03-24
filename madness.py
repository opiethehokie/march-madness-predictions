import random
import sys

import shap
import numpy as np
from deepchecks import Dataset
from deepchecks.suites import full_suite
from sklearn.metrics import log_loss, roc_curve, confusion_matrix, auc, accuracy_score, classification_report

from db.kaggle import (game_data, read_predictions, write_predictions, team_id_mapping, team_seed_mapping,
                       championship_pairings, possible_tourney_matchups)
from ml.training import linear_model, embedding_model, neural_network_model
from ml.wrangling import prepare_data
from ml.postprocessing import (override_final_predictions, average_predictions, average_prediction_probas, significance_test,
                               confidence_intervals, effect_size, statistical_power)
from simulations.bracket import simulate_tourney


random_state = 4298
random.seed(random_state)
np.random.seed(random_state)


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2019

    start_year = 2009
    start_day = 45
    check_confidence = False
    save_predictions = False
    run_simulations = False
    explain_features = False
    deepcheck = False

    # ETL

    games = game_data()
    predict_matchups, future_games = possible_tourney_matchups(predict_year)

    X_train, X_test, X_predict, y_train, y_test, cv = prepare_data(games, future_games, start_day, start_year, predict_year)

    # ML

    classification_models = [linear_model(X_train, y_train, cv=cv, rs=random_state, tune=False),
                             embedding_model(X_train, y_train, cv=cv, rs=random_state, tune=False),
                             neural_network_model(X_train, y_train, rs=random_state, tune=False, fit=False)
                            ]

    if X_test.size > 0:
        results = average_predictions([], classification_models, X_test)
        result_probas = average_prediction_probas([], classification_models, X_test)
        print('Test accuracy: %f' % accuracy_score(y_test, results))
        print('Test confustion matrix:\n', confusion_matrix(y_test, results))
        print('Test classification report:\n', classification_report(y_test, results))
        fp_rates, tp_rates, _ = roc_curve(y_test, results)
        print('Test AUC: %f' % auc(fp_rates, tp_rates))
        print('Test log loss: %f' % log_loss(y_test, result_probas))

    if save_predictions:
        prediction_probas = average_prediction_probas([], classification_models, X_predict)
        slots = championship_pairings()
        seeds = team_seed_mapping()
        write_predictions(predict_matchups, prediction_probas)
        write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 0), '-0')
        write_predictions(predict_matchups, override_final_predictions(slots, seeds, predict_matchups, prediction_probas, 1), '-1')

    # data sci

    if check_confidence and X_test.size > 0:
        model1_results = []
        model2_results = []
        for predict_year in [2015, 2016, 2017, 2018, 2019]:
            _, future_games = possible_tourney_matchups(predict_year)
            X_train, X_test, _, y_train, y_test, _ = prepare_data(games, future_games, start_day, start_year, predict_year)
            for rs in np.random.randint(0, 1000, 6):
                np.random.seed(rs)
                classification_y_train = [1 if yi > 0 else 0 for yi in y_train]
                classification_y_test = [1 if yi > 0 else 0 for yi in y_test]
                model1 = linear_model(X_train, classification_y_train, rs=rs)
                model1_results.append(log_loss(classification_y_test, model1.predict_proba(X_test)[:, 1]))
                model2 = embedding_model(X_train, classification_y_train, rs=rs)
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
        for model in classification_models:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 10))
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values) #TODO if high in training but not test then contributing to overfitting, if high in test but not training then contributing to generalization - remove if overfitting

    if deepcheck and X_test.size > 0:
        train_ds = Dataset.from_numpy(np.concatenate([X_train, y_train], axis=1))
        test_ds = Dataset.from_numpy(np.concatenate([X_test, y_test], axis=1))
        model = linear_model(X_train, classification_y_train, cv, random_state)
        suite = full_suite()
        suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)

    if run_simulations:
        simulate_tourney(team_id_mapping(), read_predictions(), predict_year)
