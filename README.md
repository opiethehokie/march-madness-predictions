# Machine Learning March Madness Predictions

Based on https://www.kaggle.com/c/mens-machine-learning-competition-2018 format.

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
