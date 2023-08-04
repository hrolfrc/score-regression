"""
The ScoreRegression and ScoreRegressionCV classifiers

"""
import functools

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def predict(X, w):
    return np.sum(np.multiply(X, w), 1)


def objective(X, y, w):
    """ Find the auc of the weights and candidate vertex.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        w : vetted weights

        Examples:
            >>> from sklearn.datasets import make_classification

            # Make a classification problem
            With 3 features in X, we pass w [1, -1] and the candidate 1
            >>> X_d, y_d = make_classification(n_samples=10, n_features=3, n_informative=2, n_redundant=1, n_repeated=0, n_classes=2, random_state=42)
            >>> auc_d = objective(X_d, y_d, [1, -1])
            >>> np.round(auc_d, 2)
            0.08

    """
    try:
        auc = roc_auc_score(y_true=y, y_score=predict(X, w))
    except ValueError:
        auc = 0

    return -auc


def opt_auc(X, y):
    """Find the weight that maximizes auc.

    Negate the auc from hv_candidate as hv_candidate_min
    because we are minimizing.  Then negate the function value
    upon return.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector

    Examples:
        >>> from sklearn.datasets import make_classification

        # Make a classification problem
        With 3 features in X, we pass w [1, -1] and the candidate 1
        >>> X_d, y_d = make_classification(
        ...    n_samples=10,
        ...    n_features=3,
        ...    n_informative=2,
        ...    n_redundant=1,
        ...    n_classes=2,
        ...    random_state=8
        ... )

        >>> y_d
        array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])

        >>> auc_d, opt_w_d = opt_auc(X_d, y_d)
        >>> np.round(auc_d, 2)
        0.2

        >>> np.round(opt_w_d, 2)
        -1.06

        >>> v = [1, -1] + [opt_w_d]
        >>> np.round(v, 2).tolist()
        [1.0, -1.0, -1.06]

        >>> y_score = np.sum(X_d * v, axis=1)
        >>> round(roc_auc_score(y_true=y_d, y_score=y_score), 2)
        0.16

        >>> import pprint
        >>> pprint.pprint(list(zip(y_d, y_score)))
        [(0, 5.16000807904542),
         (0, 1.616091591312219),
         (1, -2.612399203712827),
         (1, -1.0885808855579375),
         (0, 0.1260321126870083),
         (1, -2.9165243290179848),
         (1, 0.025875162884451464),
         (0, -1.3242115107804242),
         (0, -2.441252084537912),
         (1, -2.9968811104312167)]

    """
    print('starting opt_auc')
    # define the partial function for the optimization
    # of auc over a weight range.
    try:
        res = minimize(
            functools.partial(
                objective,
                X, y
            ),
            x0=np.random.default_rng().uniform(-1, 1, X.shape[1]),
            method='powell',
            options={'xtol': 1e-6, 'maxiter': 1e4, 'disp': False}
        )
    except ValueError:
        # powell can raise a ValueError, in which case
        # return auc == 0 and the initial condition, x0,
        # as a reasonable guess for w. The feature will be
        # skipped anyway because of the low auc.
        opt_fun, opt_w = 0, [0] * len(y)
        pass
    else:
        # res.x is an array, such as array([-1.05572788])
        # return a float
        opt_fun, opt_w = -res.fun, res.x[0]

    print('finished opt_auc')
    return opt_fun, opt_w


def phase_1_best_tuples(X, y):
    """ Get the best tuples according to AUC.

    Total optimizations: k(n + 1) - (1/2)k(k + 1)
    In phase 2, there are nk - k/2(k + 1) optimizations
    where n is the number of features and k = n-1.
    These should be parallelized using a queue.

    In phase 3, there are n-1 optimizations.
    These should be parallelized and follow from the phase 2
    optimizations.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector

    Returns:
        weights

        Examples:
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.linear_model import LogisticRegression
            >>> from sklearn.metrics import roc_auc_score
            >>> import numpy

            # Make a classification problem
            >>> X_d, y_d = make_classification(
            ...    n_samples=20,
            ...    n_features=5,
            ...    n_informative=2,
            ...    n_redundant=1,
            ...    n_classes=2,
            ...    hypercube=True,
            ...    random_state=8
            ... )

            ==================================================
            Get and round the logistic regression probabilities
            >>> clf = LogisticRegression(max_iter=10000)
            >>> lp = clf.fit(X_d, y_d).predict_proba(X_d)
            >>> lpr = np.round(lp, 2).tolist()

            auc can be predicted by predict or column 1 of predict_prob
            accuracy can be predicted by predict, requiring integers
            >>> auc_pp = roc_auc_score(y_true=y_d, y_score=clf.fit(X_d, y_d).predict_proba(X_d)[:, 1])
            >>> auc_p = roc_auc_score(y_true=y_d, y_score=clf.fit(X_d, y_d).predict(X_d))
            >>> np.round((auc_pp, auc_p), 3)
            array([0.9, 0.8])

            >>> len(y_d)

            >>> best_d = phase_1_best_tuples(X_d, y_d)
            >>> import pprint
            >>> pprint.pprint(best_d)
            [[0.9, [1], [0.47213661937437423]],
             [0.9, [1, 4], [0.47213661937437423, 1.999999537971335]],
             [0.75, [1, 4, 2], [0.47213661937437423, 1.999999537971335, 1.999999531661524]],
             [0.75,
              [1, 4, 2, 3],
              [0.47213661937437423,
               1.999999537971335,
               1.999999531661524,
               1.9999995385610347]],
             [0.75,
              [1, 4, 2, 3, 0],
              [0.47213661937437423,
               1.999999537971335,
               1.999999531661524,
               1.9999995385610347,
               1.9999995297688367]]]
    """
    print('starting phase_1_best_tuples')
    # feature range is the column index
    feature_range = list(range(X.shape[1]))

    # taken columns are the columns that have been chosen
    # for high auc.  Available columns are those not taken.
    taken = []
    weight = []
    best = []
    best_auc = []

    while available := set(feature_range).difference(set(taken)):
        print('available ', available)
        candidates = []

        for col in available:
            print('column is ', col)
            print('taken ', taken)
            X_c = X[:, taken + [col]]
            auc, w_c = opt_auc(X_c, y)
            # print('auc ', auc, ', w_c ', w_c)
            candidates += [(auc, col, w_c)]

        winner = sorted(candidates, reverse=True)[0]
        taken.append(winner[1])
        weight.append(winner[2])
        max_auc = winner[0]
        if best_auc and max_auc <= max(best_auc):
            taken = feature_range.copy()
        else:
            best_auc += [max_auc]
            best.append([max_auc, taken.copy(), weight.copy()])
        print('best auc ', max_auc)
        print('best columns ', winner[1])

    print('finished phase_1_best_tuples')

    return best


# noinspection PyAttributeOutsideInit
class ScoreRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, grid=(-1, 1)):
        self.grid = [grid] if isinstance(grid, int) else grid

    def fit(self, X, y):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.feature_range_ = list(range(X.shape[1]))
        self.sample_range_ = list(range(X.shape[0]))
        self.progress_ = {}

        # ===================================================
        #                   Phase 1
        #   Find the features that increase AUC.
        # ===================================================
        best = phase_1_best_tuples(X, y)
        self.progress_['phase_1'] = {}
        self.progress_['phase_1']['best_outcomes'] = best

        # get the best outcome for phase 1
        # sort on AUC first and if there is a tie, choose the
        # more parsimonious solution.
        winner = sorted(best, reverse=True)[0]
        self.progress_['phase_1']['auc'] = winner[0]
        self.progress_['phase_1']['columns'] = winner[1]
        self.progress_['phase_1']['weights'] = winner[2]

        # get the best outcome over all column subsets
        # the winning features are likely to be a subset of
        # all features.
        self.auc_ = winner[0]
        self.columns_ = winner[1]
        self.w_ = winner[2]

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
    def __init__(self, grid=(-1, 1), verbose=0):
        """ Initialize CalfCV

        Arguments:
            grid : the search grid.  Default is (-1, 1).
            verbose : 0, print nothing.  1-3 are increasingly verbose.

        """
        self.grid = grid
        self.verbose = verbose

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
            param_grid={'classifier__grid': [self.grid]},
            scoring="roc_auc",
            verbose=self.verbose
        )

        self.model_.fit(X, y)
        self.is_fitted_ = True

        # "best_score_: Mean cross-validated score of the best_estimator"
        # "https://stackoverflow.com/a/50233868/12865125"
        self.best_score_ = self.model_.best_score_
        self.best_coef_ = self.model_.best_estimator_['classifier'].coef_

        if self.verbose > 0:
            print()
            print('=======================================')
            print('Objective best score', self.best_score_)
            print('Best coef_ ', self.best_coef_)
            print('Objective best params', self.model_.best_params_)

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
