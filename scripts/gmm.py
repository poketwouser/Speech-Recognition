import numpy as np
from utils import log_sum_exp, log_multivariate_normal_density

# Try k-means++ initialization
class GMM:
    def __init__(self, n_components=4, n_iter=100, tol=1e-3, min_covar=1e-3):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        
        self.weights = None
        self.means = None
        self.covars = None
        
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # Uniform weights
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        
        # Randomly from data points (will try k-means if I have time)
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices].copy()
        
        # Variance
        var = np.var(X, axis=0) + self.min_covar
        self.covars = np.tile(var, (self.n_components, 1)) # Same variance for all components
        
    def _e_step(self, X):
        # log P(X | Z)
        log_prob = log_multivariate_normal_density(X, self.means, self.covars)
        
        # log P(X, Z) = log P(X|Z) + log P(Z)
        log_joint = log_prob + np.log(self.weights)
        
        # log P(X) = log sum exp of joint over K
        log_prob_norm = log_sum_exp(log_joint, axis=1)
        
        # log P(Z|X) = log P(X, Z) - log P(X)
        log_resp = log_joint - log_prob_norm
        
        # return log P(X) mean, and log P(Z|X)
        return np.mean(log_prob_norm), log_resp
        
    def _m_step(self, X, log_resp):
        n_samples, n_features = X.shape
        
        # P(Z|X)
        resp = np.exp(log_resp) # (T, K)
        
        # Cluster Mass
        sum_lamda = np.sum(resp, axis=0) + 1e-10 # (K,)
        
        # Update weights
        self.weights = sum_lamda / n_samples
        
        # Update means
        self.means = np.dot(resp.T, X) / sum_lamda[:, np.newaxis]
        
        # Update covariances
        for c in range(self.n_components):
            diff = X - self.means[c] # (T, D)
            self.covars[c] = np.dot(resp[:, c], diff ** 2) / sum_lamda[c]
            self.covars[c] += self.min_covar
            
    def fit(self, X_seqs):
        if isinstance(X_seqs, list):
            X = np.concatenate(X_seqs, axis=0)
        else:
            X = X_seqs
            
        self._initialize_parameters(X)
        
        lower_bound = -np.inf
        for i in range(self.n_iter):
            prev_lower_bound = lower_bound
            
            # E-step
            log_prob_norm, log_resp = self._e_step(X)
            lower_bound = log_prob_norm
            
            # M-step
            self._m_step(X, log_resp)
            
            if abs(lower_bound - prev_lower_bound) < self.tol:
                print(f"GMM converged at iteration {i} with log-likelihood {lower_bound:.4f}")
                break
                
    def score(self, X):
        log_prob = log_multivariate_normal_density(X, self.means, self.covars)
        log_joint = log_prob + np.log(self.weights)
        log_prob_norm = log_sum_exp(log_joint, axis=1)
        return np.sum(log_prob_norm)
