import pytest
from sklearn.utils.estimator_checks import check_estimator

from score_regression.score_regression import ScoreRegression, ScoreRegressionCV


@pytest.mark.parametrize(
    "estimator",
    [ScoreRegression(), ScoreRegressionCV()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
