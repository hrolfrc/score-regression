""" Tests for score-regression

===============================================================
Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
License: 3-clause BSD
===============================================================

"""
import pytest
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from score_regression import ScoreRegression
from sklearn.metrics import roc_auc_score
import numpy as np


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=10,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=2,
        random_state=8
    )
    return X, y


def test_ScoreRegression(synthetic_data):
    X, y = synthetic_data
    clf = ScoreRegression()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    clf.fit(X_train, y_train)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)

    # perfect match
    assert all(x == y for x, y in zip(y_pred, y_test))

    # auc of 1 to reflect the perfect match
    assert roc_auc_score(y_true=y_test, y_score=y_pred) == 1.0

    # set the random state for reproducible results
    # assert np.round(clf.coef_, 2) == [3.43]
