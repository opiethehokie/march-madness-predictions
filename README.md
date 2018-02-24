# Machine Learning March Madness Predictions

[Blog post describing methodology](http://www.programmingopiethehokie.com/2017/01/machine-learning-for-ncaa-basketball.html)

[Blog post describing performance optimizations to my rating calculators](http://www.programmingopiethehokie.com/2017/02/machine-learning-for-ncaa-basketball.html)

## Usage

Create Conda environment (tested with 64-bit Anaconda Python 3.6 on Windows):

`conda env create -f environment.yml`

Predict tournament games for years 2012 - 2017:

`python madness.py <year>`

Probabilities for every possible tournament game are written to CSV files in the results directory. These are then used to simulate the tournament and the results are printed to the console.

## Development

Run basic unit tests:

`py.test --ignore ml/tests/test_classification.py`

Run all tests:

`py.test`

Run pylint static analysis:

`pylint madness.py && pylint ml && pylint ratings`

Generate code coverage report:

`py.test --cov-report html --cov .`

Profile section of code:

```python
from ml.wrangling import profile

@profile
def slow_func() ...
```
