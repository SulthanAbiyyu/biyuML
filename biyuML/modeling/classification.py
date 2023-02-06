from collections import Counter
import numpy as np


class KNN:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, x):
        distances = [
            (np.sum((x - x_train) ** self.p)) ** 1 / self.p for x_train in self.X_train
        ]
        k_indices = np.argsort(distances)[: self.k]  # urutin jaraknya, ambil k terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)


class NaiveBayes:
    def __init__(self, mode="gaussian"):
        self.mode = mode
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.prior = {}
        for c in self.classes:
            self.prior[c] = sum(self.y_train == c) / len(self.X_train)

        self.mean, self.var = {}, {}
        for class_ in self.classes:
            for idx, feature in enumerate(X_train[y_train == class_].T):
                if self.mode == "gaussian":
                    self.mean[(class_, idx)] = np.mean(feature)
                    self.var[(class_, idx)] = np.var(feature)
                elif self.mode == "bernoulli":
                    raise NotImplementedError
                elif self.mode == "multinomial":
                    raise NotImplementedError
                else:
                    raise ValueError("Invalid mode")

    def _gaussian(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * (var))) * np.exp(
            (-1 * ((x - mean) ** 2)) / (2 * (var))
        )

    def predict_proba(self, X_test):
        self.posterior = dict.fromkeys(self.classes, None)
        for c in self.classes:
            for idx, feature in enumerate(X_test.T):
                self.posterior[c] = self._gaussian(
                    feature, self.mean[(c, idx)], self.var[(c, idx)]
                )

    def predict(self, X_test):
        self.predict_proba(X_test)
        y_pred = []
        for data_point in range(len(X_test)):
            current_max = None
            for c in self.classes:
                if current_max is None:
                    current_max = c
                else:
                    if (
                        self.posterior[c][data_point]
                        > self.posterior[current_max][data_point]
                    ):
                        current_max = c
            y_pred.append(current_max)

        return np.array(y_pred)
