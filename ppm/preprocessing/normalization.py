import numpy as np


class MinMaxNormalization:
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data))


class ZScoreNormalization:
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray):
        return (data - np.mean(data)) / np.std(data)


class DecimalScalingNormalization:
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray):
        return data / 10 ** np.ceil(np.log10(np.abs(data)))
