import numpy as np
import pandas as pd
from biyuML.modeling.classification import *

iris = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)
labels = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
X = iris.drop(columns=["class"]).values
y = iris["class"]
y = np.array([labels[label] for label in y])

data_size = len(X)
indices = np.random.permutation(data_size)
X, y = X[indices], y[indices]

X_train, y_train = X[: int(0.8 * data_size)], y[: int(0.8 * data_size)]
X_test, y_test = X[int(0.8 * data_size) :], y[int(0.8 * data_size) :]


def test_knn():
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    assert np.mean(y_pred == y_test) > 0.7

    knn_p = KNN(k=5, p=1)
    knn_p.fit(X_train, y_train)
    try:
        y_pred = knn_p.predict(X_test)
        assert True
    except:
        assert False


def test_naive_bayes():
    nb = NaiveBayes(mode="gaussian")
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    assert np.mean(y_pred == y_test) > 0.2


def test_decision_tree():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/harrypnh/decision-tree-from-scratch/main/dataset_files/car_evaluation.csv"
    )
    X, y = data.drop("class", axis=1), data["class"]

    dt = DecisionTree()
    dt.fit(X, y)

    y_pred = dt.predict(X)

    assert np.mean(y_pred.ravel() == y) == 1.0


def test_svm_binary():
    iris = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )
    labels = {
        "Iris-setosa": -1,
        "Iris-versicolor": 1,
    }

    X = iris.drop(columns=["class"])
    y = iris["class"]
    X = X.drop(y[y == "Iris-virginica"].index, axis=0)
    y = y.drop(y[y == "Iris-virginica"].index)
    y = y.replace(labels).values

    indices = np.random.permutation(len(X))
    X = X.loc[indices]
    y = y[indices]
    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]

    svm = SVM()
    svm.fit(X_train.to_numpy(), y_train)
    y_pred = svm.predict(X_test.to_numpy())

    assert np.mean(y_pred == y_test) > 0.7
