import numpy as np


SMOOTHING = True
N_CONVOLUTIONS = 50
DISTANCE_WEIGHT = 0.1
KERNEL_3_GAUSSIAN = np.array([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]
        ])
KERNEL_5_GAUSSIAN = 1 / 256 * np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ])
