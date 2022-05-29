import random
import sys

import matplotlib
import numpy as np
import shap
from deepchecks import Dataset
from deepchecks.suites import full_suite
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             classification_report, confusion_matrix, log_loss,
                             roc_curve)

from db.kaggle import (championship_pairings, game_data,
                       possible_tourney_matchups, read_predictions,
                       team_id_mapping, team_seed_mapping, write_predictions)
from ml.postprocessing import (average_prediction_probas, average_predictions,
                               confidence_intervals, effect_size,
                               override_final_predictions, significance_test,
                               statistical_power)
from ml.training import bayesian_model, embedding_model, linear_model, neural_network_model
from ml.wrangling import prepare_data
from simulations.bracket import simulate_tourney

random_state = 4298
random.seed(random_state)
np.random.seed(random_state)


if __name__ == '__main__':

    predict_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2019

    start_year = 2009
    start_day = 45
    check_confidence = False
    save_predictions = True
    run_simulations = True
    explain_features = False
    deepcheck = False

    # ETL

    games = game_data()
    predict_matchups, future_games = possible_tourney_matchups(predict_year)

    X_train, X_test, X_predict, y_train, y_test, cv = prepare_data(games, future_games, start_day, start_year, predict_year)

    # ML

    classification_models = [linear_model(X_train, y_train, cv=cv, rs=random_state, tune=False),
                             embedding_model(X_train, y_train, cv=cv, rs=random_state, tune=False),
                             neural_network_model(X_train, y_train, rs=random_state, tune=False, fit=False),
                             bayesian_model(X_train, y_train, cv=cv, tune=False)
                            ]

    random_state = None
    random.seed(random_state)
    np.random.seed(random_state)

    if X_test.size > 0:
        results = average_predictions([], classification_models, X_test)
        result_probas = average_prediction_probas([], classification_models, X_test)
        print('Test accuracy: %f' % accuracy_score(y_test, results))
        print('Test confustion matrix:\n', confusion_matrix(y_test, results))
        print('Test classification report:\n', classification_report(y_test, results))
        fp_rates, tp_rates, _ = roc_curve(y_test, results)
        print('Test AUC: %f' % auc(fp_rates, tp_rates))
        print('Test Brier score loss: %f' % brier_score_loss(y_test, result_probas))
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
            for rs in np.random.randint(0, 1000, 10):
                random.seed(rs)
                np.random.seed(rs)
                classification_y_train = [1 if yi > 0 else 0 for yi in y_train]
                classification_y_test = [1 if yi > 0 else 0 for yi in y_test]
                model1 = linear_model(X_train, classification_y_train, rs=rs)
                model1_results.append(log_loss(classification_y_test, model1.predict_proba(X_test)[:, 1]))
                model2 = embedding_model(X_train, classification_y_train, rs=rs)
                model2_results.append(log_loss(classification_y_test, model2.predict_proba(X_test)[:, 1]))
        # rough bootstrap with approx 50 samples
        print('Models are significantly different: ', significance_test(model1_results, model2_results))
        print('Effect size: ', effect_size(model1_results, model2_results))
        print('%d samples out of %f needed to see effect' % (len(model1_results), statistical_power()))
        print('95 percent confidence intervals for model 1: ', confidence_intervals(model1_results))
        print('95 percent confidence intervals for model 2: ', confidence_intervals(model2_results))
        lower, upper = confidence_intervals(np.mean(np.array([model1_results, model2_results]), axis=0))
        print('95 percent confidence intervals for average: %f - %f' % (lower, upper))

    if explain_features and X_test.size > 0:
        matplotlib.use( 'tkagg' )
        model = linear_model(X_train, y_train, cv=cv, rs=random_state)
        # game theoretic approach to global interpretability
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, nsamples=67))
        # if high in training but not test then contributing to overfitting (should remove)
        # if high in test but not training then contributing to generalization
        shap.summary_plot(explainer.shap_values(shap.sample(X_train, nsamples=67), l1_reg='num_features(10)'), show=False)
        matplotlib.pyplot.savefig('results/shap-values-linear-train.png')
        shap.summary_plot(explainer.shap_values(X_test, l1_reg='num_features(10)'), show=False)
        matplotlib.pyplot.savefig('results/shap-values-linear-test.png')

    if deepcheck and X_test.size > 0:
        cols = [str(x) for x in range(X_train.shape[1])]
        train_ds = Dataset.from_numpy(X_train, y_train, columns=cols, cat_features=[])
        test_ds = Dataset.from_numpy(X_test, y_test, columns=cols, cat_features=[])
        model = linear_model(X_train, y_train, cv=cv, rs=random_state)
        suite = full_suite()
        result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
        result.save_as_html('results/deepchecks-linear.html')

    if run_simulations:
        simulate_tourney(team_id_mapping(), read_predictions(), predict_year)
