import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


__all__ = ['soft_threshold', 'enet_cd', 'enet_path', 'ElasticNetRegressor']


def soft_threshold(beta, reg_lambda):
    """Soft Thresholding Operator

    Applies the soft thresholding function
    ..math::
        \text{sign}(\beta)(|\beta| - \lambda)_{+}
    to beta.

    Parameters
    ----------
    beta : int or array-like
        Argument to the soft-thresholding function
    reg_lambda : int
        Threshold value. If beta is below this value, then
        this function will return zero.

    Returns
    -------
    Value after applying the soft-thresholding.
    """
    return np.sign(beta) * np.maximum(np.abs(beta) - reg_lambda, 0)


def should_stop(loss, iter_idx, tol = 1e-5):
    """Checks fractional change in deviance."""
    delta_loss = abs(loss[iter_idx] - loss[iter_idx-1])

    if delta_loss / loss[iter_idx] < tol:
        return True
    return False


def enet_cd(X, y, lambda_l1, lambda_l2=0, weight=None,
            norm_cols_X=None, tol=1e-5, max_iter=500, patience=5):
    """minimizes

    (1/2N * norm(y - X weight, 2)^2 + \
        (lambda_l1 * norm(weight, 1) + (lambda_l2/2) * norm(weight, 2)^2
    """
    # default beta is all zeros
    _, n_features = X.shape
    if weight is None:
        weight = np.zeros(n_features)

    if norm_cols_X is None:
        norm_cols_X = np.mean(X**2, axis=0)

    # value of the objective function at each iteration
    loss = np.zeros(max_iter)

    for k in range(0, max_iter):
        # the full residual used to check the value of the loss function
        residual = y - np.dot(X, weight)
        loss[k] = np.mean(residual**2)

        for j in range(0, n_features):
            # partial residual (effect of all the other co-variates
            residual = residual + X[:, j] * weight[j]

            # single variable OLS estimate
            weight_ols_j = np.mean(residual * X[:, j])

            # soft-threshold the result
            weight[j] = soft_threshold(weight_ols_j, lambda_l1)
            weight[j] /= (norm_cols_X[j] + lambda_l2)

            # restore the residual
            residual = residual - X[:, j] * weight[j]
        # end-of coefficient loop

        # check if we should stop
        if (k > patience and should_stop(loss, k ,tol)):
            break
    # end-of iteration loop

    return weight


def enet_path(X, y, lambda_path=None, n_lambda=100, lambda_ratio=1e-4,
              alpha=1, distribution='gaussian', max_iter=500, tol=1e-5,
              patience=5, scale_Xy=True):
    n_samples, n_features = X.shape

    if scale_Xy:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)

        y_scaler = StandardScaler(with_std=False)
        y = np.squeeze(y_scaler.fit_transform(y.reshape(-1, 1)))

    # if not supplied use a sequence of decreasing lambdas
    # such that the largets lambda has only one non-zero coefficient
    if lambda_path is None:
        max_lambda = (1 / n_samples) * max(np.abs(np.dot(X.T, y)))
        lambda_path = np.linspace(np.log10(max_lambda),
                                  np.log10(max_lambda * lambda_ratio),
                                  n_lambda)
        lambda_path = 10 ** lambda_path
    else:
        lambda_path = np.asarray(lambda_path)
        n_lambda = lambda_path.shape[0]

    # the coefficients for each lambda value
    weights = np.empty((n_features, n_lambda))
    intercepts = np.zeros(n_lambda)
    what = np.zeros(n_features)

    for i in range(n_lambda):
        lambda_l1 = lambda_path[i] * alpha
        lambda_l2 = lambda_path[i] * (1 - alpha)

        if distribution == 'gaussian':
            what = enet_cd(X, y, lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                           weight=what, tol=tol, max_iter=max_iter,
                           patience=patience)
        else:
            raise NotImplementedError

        if scale_Xy:
            # to get the unscaled / uncentered version of the cofficients we
            # need to apply the scale to the coefficients
            # Note: x_new = x_old * center + scale => beta_new = beta_old / center.
            weights[:, i] = what / x_scaler.scale_

            # we can always calcuale the intercept from the other coefficients
            # Note: beta0 = ybar - sum(xbar_i * beta_i)
            intercepts[i] = y_scaler.mean_ - sum(x_scaler.mean_ * weights[:, i])
        else:
            weights[:, i] = what
    # end-of lambda path

    return weights, intercepts, lambda_path


class ElasticNetRegressor(BaseEstimator, RegressorMixin):
    """minimizes

    (1/2N * norm(y - X w, 2)^2 + \
        lambda * (alpha norm(w, 1) + (1/2) * (1 - alpha) * norm(w, 2)^2
    """
    def __init__(self, lambda_path='auto', n_lambda=100,
                 lambda_ratio=1e-4, max_iter=500, tol=1e-5, patience=5,
                 scale_Xy=True, alpha=1):
        self.alpha = alpha
        self.lambda_path = lambda_path
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.scale_Xy = scale_Xy


    def fit(self, X, y):
        (coefs,
         intercepts,
         lambda_path) = enet_path(X, y,
                                  lambda_path=self.lambda_path,
                                  n_lambda=self.n_lambda,
                                  lambda_ratio=self.lambda_ratio,
                                  alpha=self.alpha,
                                  max_iter=self.max_iter,
                                  tol=self.tol,
                                  patience=self.patience,
                                  scale_Xy=self.scale_Xy)

        self.coefs_ = coefs  # [n_features, n_lambda]
        self.intercepts_ = intercepts  # [n_lambda]
        self.lambda_path_ = lambda_path # [n_lambda]

    def predict(self, X):
        # predictions for each value of lambda
        return np.dot(X, self.coefs_) + self.intercepts_
