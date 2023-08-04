"""
=============================
Plot the ROC curve
=============================

An example plot of the Receiver Operating Characteristic (ROC)
curve for SLP :class:`ScoreRegression`
on the breast cancer dataset.  We want an area under the
curve (AUC) that is near 1.

"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from score_regression import ScoreRegression

# X, y = load_breast_cancer(return_X_y=True)

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

X, y = get_small_classification()
classifier = ScoreRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

RocCurveDisplay.from_predictions(
    y_test,
    y_score[:, 1],
    name="Has breast cancer",
    color="darkorange"
)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend()
plt.show()
