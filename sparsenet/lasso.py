import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


__all__ = ['soft_threshold', 'lasso_cd', 'lasso_path', 'LassoRegressor']


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


def lasso_cd(X, y, reg_lambda, beta=None, tol=1e-5, max_iter=500, patience=5):
    # default beta is all zeros
    _, n_features = X.shape
    if beta is None:
        beta = np.zeros(n_features)

    # value of the objective function at each iteration
    loss = np.zeros(max_iter)

    for k in range(0, max_iter):
        # the full residual used to check the value of the loss function
        residual = y - np.dot(X, beta)
        loss[k] = np.mean(residual**2)

        for j in range(0, n_features):
            # partial residual (effect of all the other co-variates
            residual = residual + X[:, j] * beta[j]

            # single variable OLS estimate
            beta_ols_j = np.mean(residual * X[:, j])

            # soft-threshold the result
            beta[j] = soft_threshold(beta_ols_j, reg_lambda)

            # restore the residual
            residual = residual - X[:, j] * beta[j]
        # end-of coefficient loop

        # check if we should stop
        if (k > patience and should_stop(loss, k ,tol)):
            break
    # end-of iteration loop

    return beta


def lasso_path(X, y, lambda_path=None, n_lambda=100, lambda_ratio=1e-4,
               max_iter=500, tol=1e-5, patience=5, scale_Xy=True):
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
    betas = np.empty((n_features, n_lambda))
    intercepts = np.zeros(n_lambda)
    bhat = np.zeros(n_features)

    for i in range(n_lambda):
        bhat = lasso_cd(X, y, lambda_path[i], beta=bhat, tol=tol, max_iter=max_iter,
                        patience=patience)

        if scale_Xy:
            # to get the unscaled / uncentered version of the cofficients we
            # need to apply the scale to the coefficients
            # Note: x_new = x_old * center + scale => beta_new = beta_old / center.
            betas[:, i] = bhat / x_scaler.scale_

            # we can always calcuale the intercept from the other coefficients
            # Note: beta0 = ybar - sum(xbar_i * beta_i)
            intercepts[i] = y_scaler.mean_ - sum(x_scaler.mean_ * betas[:, i])
        else:
            betas[:, i] = bhat
    # end-of lambda path

    return betas, intercepts, lambda_path


class LassoRegressor(BaseEstimator, RegressorMixin):
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
         lambda_path) = lasso_path(X, y,
                                   lambda_path=self.lambda_path,
                                   n_lambda=self.n_lambda,
                                   lambda_ratio=self.lambda_ratio,
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
