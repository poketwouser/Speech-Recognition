import numpy as np

def log_sum_exp(a, axis=None, keepdims=True):
    a_max = np.max(a, axis=axis, keepdims=True)
    
    # All entries = -inf, a_max = -inf, a - a_max = nan
    valid = np.isfinite(a_max)
    out = np.zeros_like(a_max)

    # log sum exp where the max is valid.
    # out = a_max + log(sum(exp(a - a_max))), if a_max is valid.
    diff = a - a_max
    sum_exp = np.sum(np.exp(np.where(valid, diff, 0)), axis=axis, keepdims=True)
    out = np.where(valid, np.log(sum_exp) + a_max, -np.inf)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def log_multivariate_normal_density(X, means, covars):
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    log_prob = np.empty((n_samples, n_components))
    for c in range(n_components):
        # Diagonal covariance
        log_det = np.sum(np.log(covars[c]))
        
        inv_cov = 1.0 / covars[c]
        
        # log pdf = -0.5 * (D * log(2*pi) + log_det + sum((X - mean)^2 * inv_cov))
        diff = X - means[c]
        sq_diff = (diff ** 2) * inv_cov
        exponent = np.sum(sq_diff, axis=1)
        
        log_prob[:, c] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + exponent)
        
    return log_prob
