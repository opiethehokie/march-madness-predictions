#   Copyright 2016 Michael Peters
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


import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy

from mlxtend.plotting import plot_decision_regions
from scipy.stats import skew
from sklearn.metrics import confusion_matrix, roc_curve
from pandas.tools.plotting import scatter_matrix


#TODO https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb

#TODO http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.plotting/#category_scatter
def plot_features_scatter_matrix(df):
    scatter_matrix(df)

def print_feature_info(df):
    print(df.shape)
    print(df.describe())
    print(df.head(5))
    print(skew(df)) #TODO revisit this to see if I need to transform individual features
    print(df.cov())
    print(df.corr())

def plot_auc(actual_y, predicted_y):
    fpr, tpr, _ = roc_curve(actual_y, predicted_y)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='logistic regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

#TODO http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.plotting/#plot_confusion_matrix
def plot_confusion_matrix(actual_y, predicted_y):
    matrix = confusion_matrix(actual_y, predicted_y)
    classes = [0, 1]
    plt.figure()
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#TODO implement graphs at https://github.com/rasbt/mlxtend/blob/master/docs/sources/user_guide/classifier/StackingCVClassifier.ipynb in transformers
# and do fig.savefig(...) instead of plotting
