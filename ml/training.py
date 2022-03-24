from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adagrad
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from sklearn.pipeline import Pipeline
from skopt.searchcv import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from db.cache import save_meta_data

n_jobs = 4

scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)


def print_models(func):
    def printed_func(*args, **kwargs):
        model = func(*args, **kwargs)
        if hasattr(model, 'cv_results_'):
            print('Model func: %s' % func.__name__)
            cv_keys = ('mean_test_score', 'std_test_score', 'params')
            for r, _ in enumerate(model.cv_results_['mean_test_score']):
                print("%0.3f +/- %0.2f %r" % (model.cv_results_[cv_keys[0]][r],
                                              model.cv_results_[cv_keys[1]][r] / 2.0,
                                              model.cv_results_[cv_keys[2]][r]))
            print('CV metric: %s' % model.scorer_)
            print('Best CV hyperparams: %s' % model.best_params_)
            print('Best CV score: %.2f' % model.best_score_)
            save_meta_data(func.__name__, args[0].shape, model.best_score_, model.best_params_, kwargs['rs'])
        return model
    return printed_func


@print_models
def linear_model(X, y, cv=12, rs=None, tune=False):
    grid = {
        'engineering__n_components': Real(.7, .95),
        'classification__C': Real(1e-3, 1e-1, prior='log-uniform'),
        'classification__penalty': Categorical(['l1', 'l2'])
    }

    model = Pipeline(steps=[('engineering', PCA(random_state=rs, n_components=.95)),
                            ('classification', LogisticRegression(C=.025, solver='saga', penalty='l2', random_state=rs, n_jobs=n_jobs, max_iter=10000))
                           ])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=16)

    model.fit(X, y)
    return model


@print_models
def embedding_model(X, y, cv=12, rs=None, tune=False):
    grid = {
        'engineering__n_estimators': Integer(50, 250),
        'engineering__max_depth': Integer(3, 6),
        'classification__C': Real(1e-3, 1e-1, prior='log-uniform'),
        'classification__penalty': Categorical(['l1', 'l2'])
    }

    model = Pipeline(steps=[('engineering', RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=rs, n_jobs=n_jobs)),
                            ('classification', LogisticRegression(C=.005, solver='saga', penalty='l2', random_state=rs, n_jobs=n_jobs, max_iter=10000))])

    if tune:
        model = BayesSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=rs, n_iter=32)

    model.fit(X, y)
    return model


@print_models
def neural_network_model(X, y, cv=2, rs=None, tune=False, fit=True, X_test=None, y_test=None):
    checkpoint_file_path = 'data/mlp-checkpoint.h5'
    model_file_path = 'data/mlp.h5'

    grid = {
        'batch_size': Integer(8, 16, prior='uniform', base=2),
        'hls': Integer(int(X.shape[1]/2), int(X.shape[1])),
        'lr': Real(1e-3, 1e-1, prior='log-uniform'),
        'drop': Real(0.1, 0.5, prior='uniform')
    }

    def create_mlp(hls=28, lr=5e-3, drop=.15):
        if not fit:
            #mlp: Sequential = load_model(checkpoint_file_path)
            mlp = load_model(model_file_path)
            #print(mlp.get_config())
            #print('lr', K.eval(mlp.optimizer.lr))
        else:
            mlp = Sequential()
            mlp.add(Dense(hls, activation='relu', input_shape=(X.shape[1],)))
            mlp.add(Dropout(drop))
            mlp.add(Dense(1, activation='sigmoid'))
            mlp.compile(loss='binary_crossentropy', optimizer=Adagrad(learning_rate=lr))
        return mlp

    if tune:
        classifier = KerasClassifier(build_fn=create_mlp)
        model = BayesSearchCV(classifier, grid, cv=cv, scoring=scoring, n_jobs=1, random_state=rs, n_iter=16)
    else:
        model = create_mlp()

    if fit:
        validation_data = None
        callbacks = []
        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)
            callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=0),
                         ModelCheckpoint(checkpoint_file_path, monitor='val_loss', save_best_only=True, verbose=1)]
        model.fit(X, y, validation_data=validation_data, callbacks=callbacks, epochs=32, batch_size=8, verbose=1)
        model.best_estimator_.model.save(model_file_path) if hasattr(model, 'best_estimator_') else model.save(model_file_path)

    return model
