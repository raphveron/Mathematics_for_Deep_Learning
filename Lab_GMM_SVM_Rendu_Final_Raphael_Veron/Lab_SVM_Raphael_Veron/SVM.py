import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

class SVM:
    def __init__(self, C=1.0, lr=0.01, max_iter=1000):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def _loss(self, X, Y):
        regularizer = 0.5 * np.dot(self.w, self.w)
        error_term = sum(max(0, 1 - y * (np.dot(self.w, x) + self.b)) for x, y in zip(X, Y))
        total_loss = regularizer + self.C * error_term
        return total_loss

    def _derivative_loss(self, xi, yi):
        margin = yi * (np.dot(self.w, xi) + self.b)
        if margin < 1:
            d_w = self.w - self.C * yi * xi
            d_b = -self.C * yi
        else:
            d_w = self.w
            d_b = 0
        return d_w, d_b

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.max_iter):
            for i in range(n_samples):
                d_w, d_b = self._derivative_loss(X[i], Y[i])
                self.w -= self.lr * d_w
                self.b -= self.lr * d_b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def fit_SGD(self, X, Y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.max_iter):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(n_samples):
                d_w, d_b = self._derivative_loss(X_shuffled[i], Y_shuffled[i], self.w, self.b)

                # Update parameters
                self.w -= self.lr * d_w
                self.b -= self.lr * d_b

        return self.w, self.b

    def predict_SGD(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def solve_alpha(self, X, Y):
        N = X.shape[0]  # Nombre d'échantillons
        Y = np.array(Y).reshape(-1, 1) * 1.0  # Convertir Y en une matrice de forme (N,1)
        H = np.dot(Y * X, (Y * X).T)  # Matrice H = y_i * y_j * <x_i, x_j>

        P = H
        q = -np.ones(N)
        G = np.vstack((-np.eye(N), np.eye(N)))
        h = np.hstack((np.zeros(N), np.ones(N) * self.C))
        A = Y.reshape(1, -1)
        b = np.array([0.])
    
        # Convertir en matrices creuses pour une meilleure performance
        P = csc_matrix(H)
        G = csc_matrix(np.vstack((-np.eye(N), np.eye(N))))
        A = csc_matrix(Y.reshape(1, -1))

        # Utiliser le solveur 'osqp'
        alpha = solve_qp(P, q, G, h, A, b, solver='osqp')
        return alpha

    def calculate_w(self, X, Y, alpha):
        return np.sum((alpha * Y)[:, np.newaxis] * X, axis=0)

    def calculate_b(self, X, Y, alpha, w):
        b_sum = 0
        for i in range(len(Y)):
            b_sum += (Y[i] - np.sum(alpha * Y * np.dot(X, X[i])))
        return b_sum / len(Y)
        
    def fit_QP(self, X, Y):
        alpha = self.solve_alpha(X, Y)  # Utiliser la méthode solve_alpha pour obtenir les alphas.
        self.w = self.calculate_w(X, Y, alpha)
        self.b = self.calculate_b(X, Y, alpha, self.w)
        return self.w, self.b

    @staticmethod
    def RBF(X, x, gamma):
        # Ensure both inputs are 2D arrays, where each row represents a single data point
        X = np.atleast_2d(X)
        x = np.atleast_2d(x)
        # Compute the squared L2 distance between each pair of points
        # This can be done efficiently using broadcasting and vectorized operations
        squared_norm = np.sum((X[:, np.newaxis, :] - x[np.newaxis, :, :]) ** 2, axis=2)
        # Apply the RBF kernel function element-wise
        K = np.exp(-gamma * squared_norm)
        return K



    def solve_alpha_RBF(self, X, Y, gamma):
        N = X.shape[0]
        Y = np.array(Y).reshape(-1, 1) * 1.0
        K = self.RBF(X, X, gamma)

        P = Y * Y.T * K
        q = -np.ones(N)
        G = np.vstack((-np.eye(N), np.eye(N)))
        h = np.hstack((np.zeros(N), np.ones(N) * self.C))
        A = Y.reshape(1, -1)
        b = np.array([0.])

        P = csc_matrix(P)
        G = csc_matrix(G)
        A = csc_matrix(A)

        # Solving QP problem
        alpha = solve_qp(P, q, G, h, A, b, solver='osqp')
        return alpha

    def calculate_b_RBF(self, X, Y, alpha, gamma):
        support_vectors_idx = (alpha > 1e-5).flatten()
        support_vectors = X[support_vectors_idx]
        support_vector_labels = Y[support_vectors_idx]
        alpha_sv = alpha[support_vectors_idx]

        # Calculate the RBF kernel only between support vectors
        # Ensure to pass all three required arguments: support_vectors, support_vectors, and gamma
        K_sv = self.RBF(support_vectors, support_vectors, gamma)  # Corrected call

        # Calculate b using only support vectors
        b = support_vector_labels - np.dot(K_sv, alpha_sv * support_vector_labels)
        return np.mean(b)

    def fit_QP_RBF(self, X, Y, gamma):
        alpha = self.solve_alpha_RBF(X, Y, gamma)
        self.b = self.calculate_b_RBF(X, Y, alpha, gamma)
        self.support_vectors_ = X[alpha.flatten() > 1e-5]
        self.support_vector_labels_ = Y[alpha.flatten() > 1e-5]
        self.alpha_ = alpha[alpha.flatten() > 1e-5]
        # No self.w for kernel methods
        return self.support_vectors_, self.support_vector_labels_, self.alpha_, self.b
    
    def predict_RBF(self, X, gamma):
        if self.w is not None:
            # Linear prediction
            return np.sign(np.dot(X, self.w) + self.b)
        elif gamma is not None:
            # Kernel prediction, using support vectors and the RBF kernel
            # Ensure that you have stored self.support_vectors_, self.alpha_, and self.b from the fit_QP_RBF method
            y_predict = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                kernel_value = self.RBF(self.support_vectors_, X[i, :], gamma)
                y_predict[i] = np.sum(
                    self.alpha_ * self.support_vector_labels_ * kernel_value
                )
            return np.sign(y_predict + self.b)
        else:
            raise ValueError("The model is not fitted yet or gamma is not provided for kernel prediction.")