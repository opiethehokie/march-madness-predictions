# Machine Learning March Madness Predictions

Based on https://www.kaggle.com/c/mens-machine-learning-competition-2018 format.

The raw data is the box scores of all the college games played since 2009. We want to predict a win probability for all possible tournament games of a given season. The tournament games that are then actually played post-prediction are evaluated using the log loss metric.

## High-level Approach

I make a pass through the played games for each season, in order, engineering features for each team on each day based on a team's strength and performance up to that point. This is in some ways a time-series problem and we have to be careful not to let information from the (known) future influence model training. A train/val/test split could be past regular season games / past tournament games / other past tournament games or past games / other past games / other past tournament games.

Multiple models are trained to predict a future game based on a combination of the features of the two teams that are playing known at that point in time. This can be done as binary classification of one team winning against the other team, or regression of one team's margin of victory over the other team. The best model (or combination of models) is selected and predictions are converted to win probabilities.

## Usage

Create Conda environment (developed with 64-bit Anaconda Python 3.8.2 on Windows 10):

`conda env create -f environment.yml`

Predict tournament games for a single year 2015-2019:

`python madness.py <year>`

Probabilities for every possible tournament game are written to CSV files in the results directory. These are then used to simulate the tournament and the results are printed to the console to be used for filling out a traditional bracket.

A nice way to check your results for the most recent tournament is https://www.marksmath.org/visualization/kaggle_brackets/ (not created by me).

## Development

Run all tests:

`py.test`

Run pylint static analysis:

```
pylint madness.py
pylint ml
pylint ratings
```

Generate code coverage report:

`py.test --cov-report html --cov .`

## TODO

- way to track results (hyperparameters, model, results), maybe use git lfs, see https://towardsdatascience.com/versioning-machine-learning-experiments-vs-tracking-them-f3096a67faa1, Weights & Biases, ml-metadata
- better error analysis (checking data/feature correctness with asserts, graph NN training progress, slice by seeds/rounds to check predictions), see https://www.scikit-yb.org/en/latest/api/regressor/residuals.html, https://towardsdatascience.com/how-to-find-weaknesses-in-your-machine-learning-models-ae8bd18880a3, https://nicjac.dev/posts/identify-best-model/, https://towardsdatascience.com/the-newest-package-for-instantly-evaluating-ml-models-deepchecks-d478e1c20d04, TFDV

- data-first approach to do better sampling and remove outliers: try re-training model on samples that are close to decision boundary, for over-sampling swap first/second team features
- practice joins by including new feature from another file like conference
- location feature engineering like binary home or h/a/n (only standardize/normalize where appropriate)
- additional time-series feature engineering from https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60

- try genetic algo feature selection from https://towardsdatascience.com/evolutionary-feature-selection-for-machine-learning-7f61af2a8c12
- try Tree-embedded logistic regression from  https://towardsdatascience.com/lesser-known-data-science-techniques-you-should-add-to-your-toolkit-a96578113350
- try dynamic ensemble selection from https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/
- try supervised dimensionality reduction like LDA or PLS, or PCA with % variance kept (after standardizing)

- try post-processing calibration depending on model
- try batch normamlization for NN (see 100 page book)

