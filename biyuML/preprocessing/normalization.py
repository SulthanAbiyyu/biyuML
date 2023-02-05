import numpy as np


class MinMaxNormalization:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        self.min = np.min(data)
        self.max = np.max(data)

    def transform(self, data: np.ndarray):
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)


class ZScoreNormalization:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)


class DecimalScalingNormalization:
    def __init__(self):
        self.j = None

    def fit(self, data: np.ndarray):
        self.j = np.ceil(np.log10(np.abs(data)))

    def transform(self, data: np.ndarray):
        return data / 10**self.j

    def fit_transform(self, data: np.ndarray):
        self.fit(data)
        return self.transform(data)
