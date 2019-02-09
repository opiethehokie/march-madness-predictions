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


import numpy as np

from ml2 import transformers


def test_skew_transformer():
    data = np.array([[9, 9, 1],
                     [8, 9, 2],
                     [1, 1, 1]])
    skew = transformers.SkewnessTransformer(max_skew=.7, lmbda=.5)
    assert not np.array_equal(skew.transform(data), data)
    skew = transformers.SkewnessTransformer(max_skew=.5, lmbda=0)
    assert not np.array_equal(skew.transform(data), data)
    skew = transformers.SkewnessTransformer(max_skew=.5, lmbda=None)
    assert not np.array_equal(skew.transform(data), data)
    skew = transformers.SkewnessTransformer(max_skew=100, lmbda=None)
    assert np.array_equal(skew.transform(data), data)
