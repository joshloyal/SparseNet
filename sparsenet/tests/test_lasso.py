import numpy as np

import sparsenet as snet

from sklearn.utils import check_random_state
from sklearn.preprocessing import scale


def generate_data(n_samples = 500, n_features = 20, n_nonzero_features = 5,
                  scale_Xy = False, random_state = 123):
    random_state = check_random_state(random_state)

    if n_nonzero_features > n_features:
        raise ValueError('`n_nonzero_features must be less than `n_features`')

    # a known linear model
    beta = np.zeros(n_features)
    beta[:n_nonzero_features] = np.arange(1, 6) / 5
    intercept = 0.5

    # generate X
    mean = 3 * random_state.rand(n_features) - 1
    cov = np.ones((n_features, n_features)) * 0.5
    np.fill_diagonal(cov, 1)
    X = random_state.multivariate_normal(mean=mean, cov=cov, size=n_samples)

    # create a scale for X
    X = X * np.tile(np.arange(1, 11), 2) / 5

    y = intercept + np.dot(X, beta) + random_state.randn(n_samples)

    if scale_Xy:
        X = scale(X, copy=False)
        y = scale(y, with_std=False, copy=False)

    return X, y

def test_soft_threshold():
    """Sanity checks of the soft-thresholding function."""
    assert snet.soft_threshold(10, 100) == 0
    assert snet.soft_threshold(-10, 100) == 0
    assert snet.soft_threshold(10, 3) == 7
    assert snet.soft_threshold(-10, 3) == -7


def test_soft_threshold_array():
    """Sanity checks of the soft-thresholding function."""
    a = np.array([10, -10, 200, -200])
    np.testing.assert_allclose(snet.soft_threshold(a, 100),
                               np.array([0, 0, 100, -100]))
    np.testing.assert_allclose(snet.soft_threshold(a, 3),
                               np.array([7, -7, 197, -197]))


def test_lasso_cd():
    X, y = generate_data()

    betas, intercepts, alphas = snet.lasso_path(X, y)

    # the first should only have one non-zero element
    assert np.sum(betas[:, 0] != 0) == 1
