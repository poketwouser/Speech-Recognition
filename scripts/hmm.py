import numpy as np
from utils import log_sum_exp, log_multivariate_normal_density

# Partner with feature extraction
class HMM:
    def __init__(self, n_states=5, n_iter=50, tol=1e-3, min_covar=1e-3):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        
        # log_pi, log_A, means, covars
        self.log_pi = None
        self.log_A = None
        self.means = None
        self.covars = None

    def _initialize_parameters(self, X_seqs):
        D = X_seqs[0].shape[1]
        
        # log_pi
        pi = np.zeros(self.n_states)
        pi[0] = 1.0
        self.log_pi = np.full(self.n_states, -np.inf)
        self.log_pi[0] = 0.0
        
        # log_A
        A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i == self.n_states - 1:
                A[i, i] = 1.0
            else:
                A[i, i] = 0.5
                A[i, i + 1] = 0.5
        
        self.log_A = np.log(A + 1e-10) 
        self.log_A[A == 0] = -np.inf
        
        # Means & Covars: divide each sequence into n_states equal segments
        # Partner with feature extraction
        state_data = [[] for _ in range(self.n_states)]
        
        for X in X_seqs:
            T = X.shape[0]
            chunk_size = T / self.n_states
            for i in range(self.n_states):
                start = int(i * chunk_size)
                if i < self.n_states - 1:
                    end = int((i + 1) * chunk_size)
                else:
                    end = T
                if start < end:
                    state_data[i].append(X[start:end])
                else:
                    if start < T:
                        state_data[i].append(X[start:start+1])
                    else:
                        state_data[i].append(np.zeros((1, D)))
                    
        self.means = np.zeros((self.n_states, D))
        self.covars = np.zeros((self.n_states, D))
        
        for i in range(self.n_states):
            data_i = np.vstack(state_data[i])
            self.means[i] = np.mean(data_i, axis=0)
            self.covars[i] = np.var(data_i, axis=0) + self.min_covar

    def _forward(self, log_prob):
        T, N = log_prob.shape
        log_alpha = np.empty((T, N))
        
        log_alpha[0] = self.log_pi + log_prob[0]
        
        for t in range(1, T):
            work = log_alpha[t - 1][:, np.newaxis] + self.log_A
            log_alpha[t] = log_sum_exp(work, axis=0) + log_prob[t]
            
        return log_alpha

    def _backward(self, log_prob):
        T, N = log_prob.shape
        log_beta = np.empty((T, N))
        
        log_beta[T - 1] = 0.0
        
        for t in range(T - 2, -1, -1):
            work = self.log_A + log_prob[t + 1] + log_beta[t + 1]
            log_beta[t] = log_sum_exp(work, axis=1)
            
        return log_beta

    def _e_step(self, X_seqs):
        hmm_params = {
            'log_gamma_sum': np.full(self.n_states, -np.inf),
            'log_epsilon_xi_sum': np.full((self.n_states, self.n_states), -np.inf),
            'sum_gamma_X': np.zeros((self.n_states, X_seqs[0].shape[1])),
            'sum_gamma_X2': np.zeros((self.n_states, X_seqs[0].shape[1])),
            'sum_gamma': np.zeros(self.n_states)
        }
        
        total_log_likelihood = 0.0
        
        for X in X_seqs:
            T = X.shape[0]
            log_prob = log_multivariate_normal_density(X, self.means, self.covars)
            
            log_alpha = self._forward(log_prob)
            log_beta = self._backward(log_prob)
            
            log_L = log_sum_exp(log_alpha[T - 1])
            total_log_likelihood += log_L
            
            log_gamma = log_alpha + log_beta - log_L
            gamma = np.exp(log_gamma)
            
            log_epsilon_xi = log_alpha[:-1, :, np.newaxis] + self.log_A + log_prob[1:, np.newaxis, :] + log_beta[1:, np.newaxis, :] - log_L
            
            hmm_params['log_gamma_sum'] = np.logaddexp(hmm_params['log_gamma_sum'], log_sum_exp(log_gamma[:-1], axis=0))
            hmm_params['log_epsilon_xi_sum'] = np.logaddexp(hmm_params['log_epsilon_xi_sum'], log_sum_exp(log_epsilon_xi, axis=0))
            
            hmm_params['sum_gamma_X'] += np.dot(gamma.T, X)
            hmm_params['sum_gamma_X2'] += np.dot(gamma.T, X ** 2)
            hmm_params['sum_gamma'] += np.sum(gamma, axis=0)
            
        return total_log_likelihood, hmm_params

    def _m_step(self, hmm_params):
        self.log_A = hmm_params['log_epsilon_xi_sum'] - hmm_params['log_gamma_sum'][:, np.newaxis]
        
        self.log_A[np.isnan(self.log_A)] = -np.inf
        mask = np.ones((self.n_states, self.n_states), dtype=bool)
        for i in range(self.n_states):
            mask[i, i] = False
            if i < self.n_states - 1:
                mask[i, i + 1] = False
        self.log_A[mask] = -np.inf

        A = np.exp(self.log_A)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        A = A / row_sums
        self.log_A = np.log(A + 1e-10)
        self.log_A[A == 0] = -np.inf
        
        sum_gamma = hmm_params['sum_gamma']
        sum_gamma[sum_gamma < 1e-10] = 1e-10
        
        self.means = hmm_params['sum_gamma_X'] / sum_gamma[:, np.newaxis]
        self.covars = hmm_params['sum_gamma_X2'] / sum_gamma[:, np.newaxis] - self.means ** 2
        self.covars += self.min_covar
        
    def fit(self, X_seqs):
        self._initialize_parameters(X_seqs)
        
        prev_log_L = -np.inf
        for i in range(self.n_iter):
            log_L, hmm_params = self._e_step(X_seqs)
            self._m_step(hmm_params)
            
            if abs(log_L - prev_log_L) < self.tol:
                print(f"HMM converged at iteration {i} with log-likelihood {log_L:.4f}")
                break
            prev_log_L = log_L

    def score(self, X):
        T = X.shape[0]
        log_prob = log_multivariate_normal_density(X, self.means, self.covars)
        log_alpha = self._forward(log_prob)
        return log_sum_exp(log_alpha[T - 1])
