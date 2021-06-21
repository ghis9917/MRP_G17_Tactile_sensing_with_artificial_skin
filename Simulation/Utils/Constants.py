import numpy as np

DATASET_VERSION = 11
N_SIMULATIONS = 10000
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

# INPUT VALUES
MAX_VELOCITY = 1
MAX_FRAMES = 30
MAX_FORCE = 1000
THRESHOLD_PRESS = 3
THRESHOLD_DANGEROUS = 90.5*1000
MAX_OFFSET = 10
MAX_NOISE = 10
BIG_SHAPES = ['circle blur.png', 'circle no blur.png', 'hand.png', 'hand1.png', 'hand2.png', 'fist1.png']
SMALL_SHAPES = ['circle_blur_small.png', 'circle_no_blur_small.png', 'hand_small.png']
BIG_STATIC_SHAPES = ['band.png']
SMALL_STATIC_SHAPES = ['string.png']
NEAREST_NEIGHBOUR_THRESHOLD = 100


# FEM
LAYERS = 3
SIZE = 0.2  # m - Side of square

