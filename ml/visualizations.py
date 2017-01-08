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

import matplotlib.pyplot as plt
import numpy

from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix, roc_curve


# additional ideas - https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb

def plot_scatter_matrix(df):
    scatter_matrix(df)
    plt.savefig('results/scatter.png')

def plot_auc(actual_y, predicted_y):
    fpr, tpr, _ = roc_curve(actual_y, predicted_y)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='stacked logistic regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('results/auc.png')

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
    plt.savefig('results/confusion.png')
