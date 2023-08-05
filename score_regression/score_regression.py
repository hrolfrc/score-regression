"""
The ScoreRegression and ScoreRegressionCV classifiers

"""
import functools
import multiprocessing

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def predict(X, w):
    return np.sum(np.multiply(X, w), 1)


# =============================================================
#   Phase 2, weight refinement
# =============================================================


def objective_phase_2(X, y, weight):
    """ Find the auc of the weights and candidate vertex.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        weight : vetted weights

    """
    # w_c is a float array.  Extract the float and make it a list
    try:
        auc = roc_auc_score(y_true=y, y_score=predict(X, weight))
    except ValueError:
        auc = 0

    return -auc


def opt_auc_phase_2(X, y, weight):
    """Find the weight that maximizes auc.


    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        weight : vetted weights

    """
    # define the partial function for the optimization
    # of auc over a weight range.
    try:
        res = minimize(
            functools.partial(
                objective_phase_2,
                X, y
            ),
            x0=np.add(weight, np.random.default_rng().uniform(-1, 1, len(weight))),
            method='Nelder-Mead',
            options={'maxiter': 1e4, 'disp': False}
        )
    except ValueError:
        # powell can raise a ValueError, in which case
        # return auc == 0 and the initial condition, x0,
        # as a reasonable guess for w. The feature will be
        # skipped anyway because of the low auc.
        opt_fun, opt_w = 0, [0] * X.shape[1]
        pass
    else:
        # res.x is an array, such as array([-1.05572788])
        # return a float
        opt_fun, opt_w = -res.fun, res.x
    return opt_fun, opt_w


# =============================================================
#   Phase 1, sequential weight optimization
# =============================================================

def objective(X, y, weight, w_c):
    """ Find the auc of the weights and candidate vertex.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        weight : vetted weights
        w_c : candidate weight

    """
    # w_c is a float array.  Extract the float and make it a list
    try:
        auc = roc_auc_score(y_true=y, y_score=predict(X, weight + w_c.tolist()))
    except ValueError:
        auc = 0

    return -auc


def opt_auc(X, y, weight):
    """Find the weight that maximizes auc.

    Negate the auc from hv_candidate as hv_candidate_min
    because we are minimizing.  Then negate the function value
    upon return.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        weight : vetted weights

    """
    try:
        res = minimize(
            functools.partial(
                objective,
                X, y, weight
            ),
            x0=0,
            method='powell',
            options={'xtol': 1e-6, 'maxiter': 1e4, 'disp': False}
        )
    except ValueError:
        # powell can raise a ValueError, in which case
        # return auc == 0 and the initial condition, x0,
        # as a reasonable guess for w. The feature will be
        # skipped anyway because of the low auc.
        opt_fun, opt_w = 0, 0
        pass
    else:
        # res.x is an array, such as array([-1.05572788])
        # return a float
        opt_fun, opt_w = -res.fun, res.x[0]
    return opt_fun, opt_w


def fit_mp_helper(X, y, weight, taken, col):
    X_c = X[:, taken + [col]]

    # we are going to be optimizing one position for col
    assert X_c.shape[1] - 1 == len(weight)

    auc, w_c = opt_auc(X_c, y, weight)
    return auc, col, w_c


def fit_mp(X, y, weight, taken, available):
    """ fit multiprocess

    """
    f = functools.partial(fit_mp_helper, X, y, weight, taken)
    candidates = []
    with multiprocessing.Pool(processes=40, maxtasksperchild=1000) as pool:
        iter_obj = pool.imap_unordered(f, available)
        while True:
            try:
                # get the next result and abort if there is a timeout.
                # "Also if chunksize is 1 then the next() method of the iterator returned by the
                # imap() method has an optional timeout parameter: next(timeout) will raise
                # multiprocessing.TimeoutError if the result cannot be returned within timeout seconds."
                result = iter_obj.next(timeout=5)
            except StopIteration:
                break
            except multiprocessing.TimeoutError:
                print("Timeout exceeding 5 seconds.  Skipping fit...")
            else:
                if result:
                    candidates.append(result)
    return candidates


def phase_1_best_tuples_mp(X, y):
    """ Get the best tuples according to AUC.

    """
    # feature range is the column index
    feature_range = list(range(X.shape[1]))

    # taken columns are the columns that have been chosen
    # for high auc.  Available columns are those not taken.
    taken = []
    weight = []
    best = []
    best_auc = []

    while available := set(feature_range).difference(set(taken)):
        candidates = fit_mp(X, y, weight, taken, available)
        winner = sorted(candidates, reverse=True)[0]
        taken.append(winner[1])
        weight.append(winner[2])
        max_auc = winner[0]
        if best_auc and max_auc <= max(best_auc):
            taken = feature_range.copy()
        else:
            best_auc += [max_auc]
            best.append([max_auc, taken.copy(), weight.copy()])
    return best


# noinspection PyAttributeOutsideInit
class ScoreRegression(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.feature_range_ = list(range(X.shape[1]))
        self.sample_range_ = list(range(X.shape[0]))
        self.progress_ = {}

        # ===================================================
        #                   Phase 1
        #
        #   Find the features and weights that increase AUC.
        # ===================================================
        best = phase_1_best_tuples_mp(X, y)
        winner = sorted(best, reverse=True)[0]
        self.progress_ = {'phase_1': {}}
        self.progress_['phase_1']['best_outcomes'] = best
        self.progress_['phase_1']['auc'] = winner[0]
        self.progress_['phase_1']['columns'] = winner[1]
        self.progress_['phase_1']['weights'] = winner[2]

        self.auc_ = winner[0]
        self.columns_ = winner[1].copy()
        self.w_ = winner[2].copy()

        # ===================================================
        #                   Phase 2
        #   Optimize over the most important features
        # ===================================================

        auc, weights = opt_auc_phase_2(X[:, self.columns_], y, self.w_)
        if auc > self.auc_ and not np.isclose([auc], [self.auc_]):
            print('Phase 2 improved on phase 1: ', auc, ' > ', self.auc_)
            self.w_ = weights.copy()

            self.progress_['phase_2'] = {}
            self.progress_['phase_2']['auc'] = auc
            self.progress_['phase_2']['columns'] = self.columns_
            self.progress_['phase_2']['weights'] = weights
        else:
            print('No improvement from phase 2, keeping phase 1 answers.')

        self.is_fitted_ = True
        self.coef_ = self.w_
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = np.array(
            minmax_scale(
                predict(X[:, self.columns_], self.w_),
                feature_range=(-1, 1)
            )
        )
        return scores

    def predict(self, X):
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        if len(self.classes_) < 2:
            y_class = self.y_
        else:
            # and convert to [0, 1] classes.
            y_class = np.heaviside(self.decision_function(X), 0).astype(int)
            # get the class labels
            y_class = [self.classes_[x] for x in y_class]
        return np.array(y_class)

    def predict_proba(self, X):
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)
        y_proba = np.array(
            minmax_scale(
                self.decision_function(X),
                feature_range=(0, 1)
            )
        )
        class_prob = np.column_stack((1 - y_proba, y_proba))
        return class_prob

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }


# noinspection PyAttributeOutsideInit
class ScoreRegressionCV(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)

        self.pipeline_ = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('classifier', ScoreRegression())
            ]
        )

        # setting n_jobs somehow cause y_true to have one class.
        # The error will look like a cv problem, but the "culprit is
        # Python’s multiprocessing that does fork without exec"
        # Do not set n_jobs in GridSearchCV until resolved.
        # https://scikit-learn.org/stable/faq.html#id27
        self.model_ = GridSearchCV(
            estimator=self.pipeline_,
            param_grid={},
            scoring="roc_auc"
        )

        self.model_.fit(X, y)
        self.is_fitted_ = True

        # "best_score_: Mean cross-validated score of the best_estimator"
        # "https://stackoverflow.com/a/50233868/12865125"
        self.best_score_ = self.model_.best_score_
        self.best_coef_ = self.model_.best_estimator_['classifier'].coef_

        # if self.verbose_ > 0:
        #     print()
        #     print('=======================================')
        #     print('Objective best score', self.best_score_)
        #     print('Best coef_ ', self.best_coef_)
        #     print('Objective best params', self.model_.best_params_)

        return self

    def decision_function(self, X):
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict_proba(X)

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }
