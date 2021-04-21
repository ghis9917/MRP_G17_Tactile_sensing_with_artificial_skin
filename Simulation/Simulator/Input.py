import numpy as np
from Shapes import Shape


class Input:
    def __init__(self, shape, vel, frames):
        self.shape: Shape = shape
        self.vel: np.ndarray = vel
        self.frames = frames

    def update_frame(self):
        self.shape.set_traslated_map(self.vel)
