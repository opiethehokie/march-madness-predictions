# Machine Learning March Madness Predictions

https://www.kaggle.com/c/mens-machine-learning-competition-2018

## Usage

Create Conda environment (tested with 64-bit Anaconda Python 3.6 on Windows):

`conda env create -f environment.yml`

Predict tournament games for years 2013+:

`python madness.py <year>`

Delete 'data/*cache.csv' files when switching years.

Probabilities for every possible tournament game are written to CSV files in the results directory. These are then used to simulate the tournament and the results are printed to the console to be used for filling out a traditional bracket.

A nice way to check your results for the most recent tournament is [https://www.marksmath.org/visualization/kaggle_brackets/]() (not created by me).

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
