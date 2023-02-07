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


# TODO: Implement pruning
class DecisionTree:
    def __init__(self):
        self.tree = None

    def _gini_impurity(self, y):
        _class, counts = np.unique(y, return_counts=True)
        return 1 - np.sum(
            [(counts[i] / np.sum(counts)) ** 2 for i in range(len(_class))]
        )

    def _gini_split(self, X, y, feature):
        _feature, counts = np.unique(X, return_counts=True)
        gini_split = np.sum(
            [
                (counts[i] / np.sum(counts))
                * self._gini_impurity(y[(X[feature] == _feature[i]).dropna()])
                for i in range(len(_feature))
            ]
        )
        return gini_split

    def _grow_tree(self, X, X_ori, y, parent_class=None):
        if len(np.unique(y)) <= 1:
            return np.unique(y)
        elif len(X) == 0:
            return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
        elif len(X.columns) == 0:
            return parent_class
        else:
            parent_class = np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
            gini_splits = [self._gini_split(X_ori, y, feature) for feature in X.columns]
            best_feature = X.columns[np.argmin(gini_splits)]
            tree = {best_feature: {}}
            for value in np.unique(X[best_feature]):
                sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
                sub_y = y[X[best_feature] == value]
                subtree = self._grow_tree(sub_X, X_ori, sub_y, parent_class)
                tree[best_feature][value] = subtree
            return tree

    def fit(self, X, y):
        self.tree = self._grow_tree(X, X, y)

    def _predict(self, X_test, tree=None):
        if tree is None:
            tree = self.tree
        if not isinstance(X_test, dict):
            X_test = X_test.to_dict()
        for key in list(X_test.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][X_test[key]]
                except:
                    return
                result = tree[key][X_test[key]]

                if isinstance(result, dict):
                    return self._predict(X_test, tree=result)
                else:
                    return result

    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            y_pred.append(self._predict(X_test.T[i]))
        return np.array(y_pred)


class SVM:
    def __init__(self, mode="binary"):
        self.mode = mode

    def _loss_binary(self, w, X, y, reg_rate=10000, lr=0.000001):
        for idx, x in enumerate(X):
            dist = 1 - (y[idx] * np.dot(x, w))
            grad = np.zeros(len(w))

            if max(0, dist) == 0:
                grad = w
            else:
                grad = w - (reg_rate * y[idx] * x)

            w = w - (lr * grad)
        return w

    def fit(self, X, y, epoch=1000):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")

        w = np.zeros(X.shape[1])
        for _ in range(epoch):
            if self.mode == "binary":
                self.w = self._loss_binary(w, X, y)
            else:
                raise NotImplementedError("Only binary classification is implemented")

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        return np.sign(np.dot(X, self.w))
