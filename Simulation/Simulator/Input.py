import numpy as np
from Shapes import Shape
from Classes import Class


class Input:
    def __init__(self, shape, vel, frames, simulation_class: Class):
        self.shape: Shape = shape
        self.vel: np.ndarray = vel
        self.frames = frames
        self.simulation_class = simulation_class

    def update_frame(self, vel = None):
        if vel is None:
            self.shape.set_traslated_map(self.vel)
        else:
            self.shape.set_traslated_map(vel)

