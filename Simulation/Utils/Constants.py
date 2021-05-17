import numpy as np

DATASET_VERSION = 7
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
MAX_VELOCITY = 5
MAX_FRAMES = 30
MAX_FORCE = 15
THRESHOLD_PRESS = 3
THRESHOLD_DANGEROUS = 9.05
MAX_OFFSET = 10
MAX_NOISE = 10
SHAPES = ['circle blur.png', 'circle_blur_small.png', 'circle no blur.png', 'circle_no_blur_small.png', 'hand.png', 'hand_small.png']

NEAREST_NEIGHBOUR_THRESHOLD = 100

def BIG(val: str) -> bool:
    if val == 'circle blur.png':
        return True
    elif val == 'circle no blur.png':
        return True
    elif val == 'hand.png':
        return True
    elif val == 'circle_blur_small.png':
        return False
    elif val == 'circle_no_blur_small.png':
        return False
    elif val == 'hand_small.png':
        return False

