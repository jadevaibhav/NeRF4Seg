import numpy as np


def segmentation3d(result):
    z = np.zeros(shape=(59, result.shape[0], result.shape[1]))
    for i in range(0, 59):
        z[i] = result == i
    return z
