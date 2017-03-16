# Machine Learning March Madness Predictions

[![Build Status](https://travis-ci.org/opiethehokie/march-madness-predictions.svg?branch=master)](https://travis-ci.org/opiethehokie/march-madness-predictions)

[Blog post describing methodology](http://www.programmingopiethehokie.com/2017/01/machine-learning-for-ncaa-basketball.html)

[Blog post describing performance optimizations to my rating calculators](http://www.programmingopiethehokie.com/2017/02/machine-learning-for-ncaa-basketball.html)

## Usage

Build Docker image:

`docker build -t madness .`

Predict tournament games for years 2012 - 2017:

`docker run --rm -it -v $PWD:/workdir madness python3 madness.py <year>`

## Development

Run basic unit tests:

`docker run --rm -it -v $PWD:/workdir madness py.test --cov=. --ignore ml/tests/test_classification.py`

Run all tests:

`docker run --rm -it -v $PWD:/workdir madness py.test --cov=.`

Run pylint static analysis:

`docker run --rm -it -v $PWD:/workdir madness pylint madness.py && pylint ml && pylint ratings`

Profile section of code:

```python
from ml.wrangling import profile

@profile
def slow_func() ...
```


