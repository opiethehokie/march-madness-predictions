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


import os

import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ml2 import postprocessing


np.random.seed(42)


def test_confidence_intervals():
    # load dataset
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(this_dir + '/pima-indians-diabetes.csv', header=None)
    values = data.values
    # configure bootstrap
    n_iterations = 100
    n_size = int(len(data) * 0.50)
    # run bootstrap
    stats = list()
    for _ in range(n_iterations):
        # prepare train and test sets
        train = resample(values, n_samples=n_size)
        test = np.array([x for x in values if x.tolist() not in train.tolist()])
        # fit model
        model = DecisionTreeClassifier()
        model.fit(train[:, :-1], train[:, -1])
        # evaluate model
        predictions = model.predict(test[:, :-1])
        score = accuracy_score(test[:, -1], predictions)
        stats.append(score)
    # confidence intervals
    lower, upper = postprocessing.confidence_intervals(stats)
    assert int(lower*100) == 64
    assert int(upper*100) == 72

def test_significance():
    vals1 = np.random.normal(50, 10, 1000)
    vals2 = np.random.normal(60, 10, 1000)
    assert postprocessing.significance_test(vals1, vals2)
    assert not postprocessing.significance_test(vals1, vals1)
    vals3 = np.random.normal(50, 1, 1000)
    assert postprocessing.significance_test(vals1, vals3)
    vals4 = np.random.randint(50, 60, 1000)
    vals5 = np.random.randint(55, 65, 1000)
    assert postprocessing.significance_test(vals4, vals5)
    assert not postprocessing.significance_test(vals4, vals4)
