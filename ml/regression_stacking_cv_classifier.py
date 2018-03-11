#   Copyright 2018 Michael Peters
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


from mlxtend.classifier import StackingCVClassifier
from mlxtend.externals.name_estimators import _name_estimators
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.model_selection import check_cv
import numpy as np

# mostly copied from https://github.com/rasbt/mlxtend/blob/master/mlxtend/classifier/stacking_cv_classification.py
class RegressionStackingCVClassifier(StackingCVClassifier):

    def __init__(self,
                 regressors,
                 meta_classifier,
                 to_class_func,
                 cv=2,
                 stratify=True,
                 shuffle=True,
                 verbose=0,
                 store_train_meta_features=False,
                 refit=True):

        self.regressors = regressors
        self.meta_classifier = meta_classifier
        self.to_class_func = to_class_func
        self.named_regressors = {key: value for key,
                                 value in _name_estimators(regressors)}
        self.named_meta_classifier = {
            'meta-%s' % key: value for key, value in _name_estimators([meta_classifier])}
        self.use_probas = False
        self.verbose = verbose
        self.cv = cv
        self.use_features_in_secondary = False
        self.stratify = stratify
        self.shuffle = shuffle
        self.store_train_meta_features = store_train_meta_features
        self.refit = refit

    def fit(self, X, y, groups=None):
        if self.refit:
            self.regs_ = [clone(reg) for reg in self.regressors]
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.regs_ = self.regressors
            self.meta_clf_ = self.meta_classifier
        if self.verbose > 0:
            print("Fitting %d regressors..." % (len(self.regressors)))

        clf_y = np.fromiter((self.to_class_func(yi) for yi in y), y.dtype)

        final_cv = check_cv(self.cv, clf_y, classifier=self.stratify)
        if isinstance(self.cv, int):
            # Override shuffle parameter in case of self generated cross-validation strategy
            final_cv.shuffle = self.shuffle
        skf = list(final_cv.split(X, clf_y, groups))

        all_model_predictions = np.array([]).reshape(len(y), 0)

        for model in self.regs_:

            if self.verbose > 0:
                i = self.regs_.index(model) + 1
                print("Fitting regressor%d: %s (%d/%d)" % (i, _name_estimators((model,))[0][0], i, len(self.regs_)))

            if self.verbose > 2:
                if hasattr(model, 'verbose'):
                    model.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((model,))[0][1])

            if not self.use_probas:
                single_model_prediction = np.array([]).reshape(0, 1)
            else:
                single_model_prediction = np.array([]).reshape(0, len(set(y)))

            for num, (train_index, test_index) in enumerate(skf):

                if self.verbose > 0:
                    print("Training and fitting fold %d of %d..." % ((num + 1), final_cv.get_n_splits()))

                try:
                    model.fit(X[train_index], y[train_index])
                except TypeError as e:
                    raise TypeError(str(e) + '\nPlease check that X and y are NumPy arrays. If X and y are lists'
                                    ' of lists,\ntry passing them as numpy.array(X) and numpy.array(y).')
                except KeyError as e:
                    raise KeyError(str(e) + '\nPlease check that X and y are NumPy arrays. If X and y are pandas'
                                   ' DataFrames,\ntry passing them as X.values and y.values.')

                if not self.use_probas:
                    prediction = model.predict(X[test_index])
                    prediction = prediction.reshape(prediction.shape[0], 1)
                    print(prediction.shape)
                else:
                    prediction = model.predict_proba(X[test_index])
                single_model_prediction = np.vstack([single_model_prediction.astype(prediction.dtype), prediction])

            print('combining ', all_model_predictions.shape, single_model_prediction.shape)
            all_model_predictions = np.hstack([all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction])

        if self.store_train_meta_features:
            self.train_meta_features_ = all_model_predictions

        # We have to shuffle the labels in the same order as we generated
        # predictions during CV (we kinda shuffled them when we did
        # Stratified CV).
        # We also do the same with the features (we will need this only IF
        # use_features_in_secondary is True)
        reordered_labels = np.array([]).astype(y.dtype)
        reordered_features = np.array([]).reshape((0, X.shape[1])).astype(X.dtype)
        for train_index, test_index in skf:
            reordered_labels = np.concatenate((reordered_labels, clf_y[test_index]))
            reordered_features = np.concatenate((reordered_features, X[test_index]))

        # Fit the base models correctly this time using ALL the training set
        for model in self.regs_:
            model.fit(X, y)

        # Fit the secondary model
        if not self.use_features_in_secondary:
            self.meta_clf_.fit(all_model_predictions, reordered_labels)
        else:
            self.meta_clf_.fit(np.hstack((reordered_features, all_model_predictions)), reordered_labels)

        return self

    def get_params(self, deep=True):
        if not deep:
            return super(RegressionStackingCVClassifier, self).get_params(deep=False)
        out = self.named_regressors.copy()
        for name, step in six.iteritems(self.named_regressors):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value

        out.update(self.named_meta_classifier.copy())
        for name, step in six.iteritems(self.named_meta_classifier):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value

        for key, value in six.iteritems(super(RegressionStackingCVClassifier, self).get_params(deep=False)):
            out['%s' % key] = value

        return out

    def predict_meta_features(self, X):
        check_is_fitted(self, 'regs_')
        all_model_predictions = np.array([]).reshape(len(X), 0)
        for model in self.regs_:
            if not self.use_probas:
                single_model_prediction = model.predict(X)
                single_model_prediction = single_model_prediction.reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(X)
            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))
        return all_model_predictions

    def predict_proba(self, X):
        check_is_fitted(self, 'regs_')
        all_model_predictions = np.array([]).reshape(len(X), 0)
        for model in self.regs_:
            if not self.use_probas:
                single_model_prediction = model.predict(X)
                single_model_prediction = single_model_prediction.reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(X)
            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))
        if not self.use_features_in_secondary:
            return self.meta_clf_.predict_proba(all_model_predictions)
        return self.meta_clf_.predict_proba(np.hstack((X, all_model_predictions)))
