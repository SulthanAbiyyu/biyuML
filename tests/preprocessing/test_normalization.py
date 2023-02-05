import numpy as np
from biyuML.preprocessing.normalization import *


def test_min_max():
    data = np.array([1, 2, 3, 4, 5])
    min_max = MinMaxNormalization()
    normalized = min_max.fit_transform(data)
    assert np.array_equal(normalized, np.array([0, 0.25, 0.5, 0.75, 1]))
    assert np.min(normalized) == 0.0
    assert np.max(normalized) == 1.0


def test_z_score():
    data = np.array([1, 2, 3, 4, 5])
    z_score = ZScoreNormalization()
    normalized = z_score.fit_transform(data)
    assert np.allclose(
        normalized, np.array([-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356])
    )
    assert np.mean(normalized) == 0.0
    assert np.allclose(np.std(normalized), 1.0)


def test_decimal_scaling():
    data = np.array([1, 2, 3, 4, 5])
    decimal_scaling = DecimalScalingNormalization()
    normalized = decimal_scaling.fit_transform(data)

    assert np.max(np.max(normalized)) <= 1.0
