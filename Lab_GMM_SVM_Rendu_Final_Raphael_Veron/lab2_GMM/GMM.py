import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, K, n_runs=200):
        self.K = K
        self.n_runs = n_runs
        self.pi = None
        self.mu = None
        self.sigma = None

    def _calculate_mean_covariance(self, X, prediction):
        d = X.shape[1]
        labels = np.unique(prediction)
        self.mu = np.zeros((self.K, d))
        self.sigma = np.zeros((self.K, d, d))
        self.pi = np.zeros(self.K)

        for counter, label in enumerate(labels):
            ids = np.where(prediction == label)
            self.pi[counter] = len(ids[0]) / X.shape[0]
            self.mu[counter, :] = np.mean(X[ids], axis=0)  # Ceci devrait être un vecteur 1D
            de_meaned = X[ids] - self.mu[counter, :]
            Nk = len(ids[0])
            self.sigma[counter] = np.dot(de_meaned.T, de_meaned) / Nk

        assert np.isclose(np.sum(self.pi), 1)
        return self.mu, self.sigma, self.pi

    def _initialise_parameters(self, X):
        kmeans = KMeans(n_clusters=self.K, init="k-means++", max_iter=500, n_init=10, algorithm='lloyd')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        return self._calculate_mean_covariance(X, prediction)

    def get_params(self):
        return self.mu, self.pi, self.sigma

    def _e_step(self, X):
        N, d = X.shape
        self.gamma = np.zeros((N, self.K))

        for k in range(self.K):
            self.gamma[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k], self.sigma[k])

        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)
        return self.gamma

    def _m_step(self, X):
        N, d = X.shape
        self.pi = np.sum(self.gamma, axis=0) / N
        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis]
        self.sigma = np.zeros((self.K, d, d))

        for k in range(self.K):
            X_centered = X - self.mu[k]
            gamma_diag = np.diag(self.gamma[:, k])
            self.sigma[k] = np.dot(X_centered.T, np.dot(gamma_diag, X_centered)) / np.sum(self.gamma[:, k])

        return self.pi, self.mu, self.sigma

    def fit(self, X):
        self._initialise_parameters(X)
        for _ in range(self.n_runs):
            self._e_step(X)
            self._m_step(X)
        return self

    def predict(self, X):
        self._e_step(X)
        return np.argmax(self.gamma, axis=1)

    def predict_proba(self, X):
        post_proba = np.array([self.pi[k] * mvn.pdf(X, self.mu[k], self.sigma[k]) for k in range(self.K)]).T
        return post_proba / np.sum(post_proba, axis=1, keepdims=True)
