import numpy as np

from sklearn.preprocessing import StandardScaler


__all__ = ['soft_threshold', 'lasso_cd', 'lasso_path']


def soft_threshold(beta, alpha):
    """Soft Thresholding Operator

    Applies the soft thresholding function
    ..math::
        \text{sign}(\beta)(|\beta| - \alpha)_{+}
    to beta.

    Parameters
    ----------
    beta : int or array-like
        Argument to the soft-thresholding function
    alpha : int
        Threshold value. If beta is below this value, then
        this function will return zero.

    Returns
    -------
    Value after applying the soft-thresholding.
    """
    return np.sign(beta) * np.maximum(np.abs(beta) - alpha, 0)


def should_stop(loss, iter_idx, tol = 1e-5):
    """Checks fractional change in deviance."""
    delta_loss = abs(loss[iter_idx] - loss[iter_idx-1])

    if delta_loss / loss[iter_idx] < tol:
        return True
    return False


def lasso_cd(X, y, alpha, beta=None, tol=1e-5, max_iter=500, patience=5):
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
            beta[j] = soft_threshold(beta_ols_j, alpha)

            # restore the residual
            residual = residual - X[:, j] * beta[j]
        # end-of coefficient loop

        # check if we should stop
        if (k > patience and should_stop(loss, k ,tol)):
            break
    # end-of iteration loop

    return beta


def lasso_path(X, y, alpha=None, n_alpha=100, alpha_ratio=1e-4,
               max_iter=500, tol=1e-5, patience=5, scale_Xy=True):
    n_samples, n_features = X.shape

    if scale_Xy:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)

        y_scaler = StandardScaler(with_std=False)
        y = np.squeeze(y_scaler.fit_transform(y.reshape(-1, 1)))

    # if not supplied use a sequence of decreasing alphas
    # such that the largets lambda has only one non-zero coefficient
    if alpha is None:
        max_alpha = (1 / n_samples) * max(np.abs(np.dot(X.T, y)))
        alphas = np.linspace(np.log10(max_alpha),
                             np.log10(max_alpha * alpha_ratio),
                             n_alpha)
        alphas = 10 ** alphas

    # the coefficients for each lambda value
    betas = np.empty((n_features, n_alpha))
    intercepts = np.zeros(n_alpha)
    bhat = np.zeros(n_features)

    for i in range(n_alpha):
        bhat = lasso_cd(X, y, alphas[i], beta=bhat, tol=tol, max_iter=max_iter,
                        patience=patience)

        # to get the unscaled / uncentered version of the cofficients we
        # need to apply the scale to the coefficients
        # Note: x_new = x_old * center + scale => beta_new = beta_old / center.
        betas[:, i] = bhat / x_scaler.scale_

        # we can always calcuale the intercept from the other coefficients
        # Note: beta0 = ybar - sum(xbar_i * beta_i)
        intercepts[i] = y_scaler.mean_ - sum(x_scaler.mean_ * betas[:, i])
    # end-of alpha path

    return betas, intercepts, alpha
