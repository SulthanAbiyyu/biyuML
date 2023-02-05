import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        mean_vec = np.mean(X, axis=0)
        cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_pairs = [
            (np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))
        ]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # eig vectors
        self.components = np.array([eig_pairs[i][1] for i in range(self.n_components)])
        # eig values
        self.explained_variance = np.array(
            [eig_pairs[i][0] for i in range(self.n_components)]
        )
        self.explained_variance_ratio = self.explained_variance / np.sum(
            self.explained_variance
        )

    def transform(self, X):
        return X.dot(self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
