#   Copyright 2016-2018 Michael Peters
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


import numpy
import pandas

from scipy.stats import boxcox, skew
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        X = pandas.DataFrame(X)
        if not self.cols:
            return X.values
        return X[self.cols].values

class DebugFeatureProperties(BaseEstimator, TransformerMixin):

    def fit(self, X, _y=None):
        df = pandas.DataFrame(X)
        print(df.shape)
        print(df.head(3))
        #print(df.describe())
        #print(skew(df))
        #print(df.cov())
        #print(df.corr())
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        return X

class SkewnessTransformer(BaseEstimator, TransformerMixin):

    # lmbda 0 is log
    # lmbda .5 is square root
    # lmbda 1 is no transform
    # lmbda None is statistically tuned
    def __init__(self, max_skew=2.5, lmbda=None):
        self.max_skew = max_skew
        self.lmbda = lmbda

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        X = pandas.DataFrame(X)
        skewed_feats = X.apply(skew)
        very_skewed_feats = skewed_feats[numpy.abs(skewed_feats) > self.max_skew]
        transformed = X.copy()
        for idx in very_skewed_feats.index:
            transformed[idx] = boxcox(X[idx]+1, lmbda=self.lmbda)[0]
        return transformed.values

class DiffTransformer(BaseEstimator, TransformerMixin):

    def fit(self, _X, _y=None):
        return self

    #pylint: disable=no-self-use
    def transform(self, X):
        df = pandas.DataFrame(X)
        even_df = df.iloc[:, [i for i in range(len(df.columns)) if i%2 == 0]]
        odd_df = df.iloc[:, [i for i in range(len(df.columns)) if i%2 == 1]]
        return pandas.DataFrame(even_df.values - odd_df.values).values
