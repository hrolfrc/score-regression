.. -*- mode: rst -*-

|CircleCI|_ |ReadTheDocs|_

.. |CircleCI| image:: https://circleci.com/gh/hrolfrc/score-regression.svg?style=shield
.. _CircleCI: https://circleci.com/gh/hrolfrc/score-regression

.. |ReadTheDocs| image:: https://readthedocs.org/projects/score-regression/badge/?version=latest
.. _ReadTheDocs: https://score-regression.readthedocs.io/en/latest/?badge=latest


ScoreRegression - A classifier that maximizes AUC
============================================================

An AUC optimizing binomial classifier.

Contact
------------------
Rolf Carlson hrolfrc@gmail.com

Install
------------------
Use pip to install score_regression.

``pip install score-regression``

Introduction
------------------
This is a python implementation of a classifier that maximizes AUC.  The idea is to find the features that maximize AUC, analogous to CALF_, but relax the requirement that the weights be integers in [-1, 0, 1] and instead allow the weights to be any real number.

ScoreRegression provides classification and prediction for two classes, the binomial case.  Small to medium problems are supported.  This is research code and a work in progress.

ScoreRegression is designed for use with scikit-learn_ pipelines and composite estimators.

.. _scikit-learn: https://scikit-learn.org

.. _CALF: https://www.nature.com/articles/s41598-022-09415-2

Example
------------------

.. code:: ipython2

    from score_regression import ScoreRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

Make a classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    seed = 42
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

Train the classifier
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls = ScoreRegression().fit(X_train, y_train)

Get the score on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)


.. parsed-literal::

    1.0

References
------------------
[1] Jeffries, C.D., Ford, J.R., Tilson, J.L. et al.
A greedy regression algorithm with coarse weights offers novel advantages.
Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2
