"""
This is the Calf classifier

"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def get_small_classification():
    """ Make a classification problem for visual inspection. """
    X, y = make_classification(
        n_samples=10,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=2,
        hypercube=True,
        random_state=8
    )
    return X, y


def predict(X, w):
    return np.sum(np.multiply(X, w), 1)


def hv_candidate(X, y, w, c):
    """ Find the auc of the weights and candidate vertex.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        w : vetted weights
        c : candidate weight

        Examples:
            >>> from sklearn.datasets import make_classification

            # Make a classification problem
            With 3 features in X, we pass w [1, -1] and the candidate 1
            >>> X_d, y_d = make_classification(n_samples=10, n_features=3, n_informative=2, n_redundant=1, n_repeated=0, n_classes=2, random_state=42)
            >>> auc_d = hv_candidate(X_d, y_d, [1, -1], 1)
            >>> np.round(auc_d, 2)
            0.08

    """
    assert X.shape[1] - 1 == len(w), "X or w have the wrong shape"
    y_p = predict(X, w + [c])
    try:
        auc = roc_auc_score(y_true=y, y_score=y_p)
    except ValueError:
        auc = 0
    return auc


def hv_max(X, y, w, c):
    """Find the weight in b that maximizes auc

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        w : vetted weights
        c : a list of candidate weights

    Examples:
        >>> X_d, y_d = get_small_classification()
        >>> rb_d = ScoreRegression()
        >>> kfold_d = StratifiedKFold(n_splits=5)

        # hypercube vertex approximation
        >>> auc_d, w_d = hv_max(X_d, y_d, [1, -1], [1, -1])
        >>> np.round((auc_d, w_d), 2)
        array([ 0.16, -1.  ])

        # granular approximation
        >>> import numpy
        >>> auc_d, w_d = hv_max(X_d, y_d, [1, -1], numpy.arange(-1, 1, .1))
        >>> np.round((auc_d, w_d), 2)
        array([ 0.16, -0.8 ])

        We get a higher auc by allowing the range to be off the vertices of the cube
        >>> auc_d, w_d = hv_max(X_d, y_d, [1, -1], numpy.arange(-2, 2, .1))
        >>> np.round((auc_d, w_d), 2)
        array([ 0.36, -1.9 ])

    """
    res = [(hv_candidate(X, y, w, v), v) for v in c]
    return sorted(res, reverse=True)[0]


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
def fit_hv(X, y, grid):
    """ Find the weights that best fit X using hypercube vertices

        Examples:
            Make a classification problem
            With 3 features in X, we pass w [1, -1] and the candidate 1
            >>> X_d, y_d = get_small_classification()
            >>> w_d = fit_hv(X_d, y_d, [-1, 1])
            >>> w_d
            [1, 1, 1]

    """
    feature_range = range(X.shape[1])
    w = []
    auc = []
    for i in feature_range:
        X_c = X[:, 0:i + 1]

        # granular approximation
        max_auc, w_c = hv_max(X_c, y, w, grid)

        # if the auc goes down then we skip the feature by weighting it at 0
        if auc and max_auc <= max(auc):
            w = w + [0]
        else:
            w = w + [w_c]
        auc = auc + [max_auc]
    return w


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
        self.w_ = fit_hv(X, y, grid=self.grid)
        self.is_fitted_ = True
        self.coef_ = self.w_
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = np.array(
            minmax_scale(
                predict(X, self.w_),
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
