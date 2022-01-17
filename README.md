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
