from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


class Sensor:
    def __init__(self, id_num, xs, ys, coords, offset, noise_sd, size):
        self.id = id_num
        self.coords: List[Tuple] = coords
        self.map = self.create_map(size)
        self.center = self.compute_center()
        # self.reading = 0
        # Each sensor will have a certain offset, which is negative for output that is too low
        # and positive for output that is too high. For now, the offset is a constant! This can
        # be changed to an offset function
        self.offset = offset
        # Each sensor will be subject to noise, the type of which still has to be decided upon.
        # (Gaussian noise for now) the noise variable (for now) is the standard deviation.
        self.noise_sd = noise_sd
        self.tilt = np.array([[0][0]])
        self.depth = 0

    def get_reading(self, val):
        return val + np.random.normal(0, self.noise_sd) + self.offset

    def create_map(self, size):
        map = np.zeros(shape=size)
        for coord in self.coords:
            map[coord[0], coord[1]] = 1
        return map

    # def set_reading(self, value: float) -> None:
    #     self.reading = value
    def compute_center(self):
        return [sum(ele) / len(self.coords) for ele in zip(*self.coords)]
