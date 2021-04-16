import numpy as np
from Shapes.Shapes import Shape


class Input:
    def __init__(self, shape, vel, frames):
        self.shape: Shape = shape
        self.vel: np.ndarray = vel
        self.frames = frames

    def update_frame(self):
        self.shape.update_center(self.vel)
