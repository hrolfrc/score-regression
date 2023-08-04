import pytest
from sklearn.datasets import load_iris

from score_regression import ScoreRegression


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_ScoreRegression(data):
    X, y = data
    clf = ScoreRegression()

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
