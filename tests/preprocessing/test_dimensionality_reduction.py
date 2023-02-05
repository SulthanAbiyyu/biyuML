import numpy as np
from biyuML.preprocessing.dimensionality_reduction import *


def test_pca():
    data_10d = np.random.rand(100, 10)
    pca = PCA(n_components=2)
    pca.fit(data_10d)
    data_2d = pca.transform(data_10d)
    assert data_2d.shape == (100, 2)
    assert np.allclose(np.sum(pca.explained_variance_ratio), 1.0)
