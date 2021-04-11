import numpy as np


class Sensor:
    def __init__(self, id_num, x, y, offset, noise_sd):
        self.id = id_num
        self.x = x
        self.y = y
        # Each sensor will have a certain offset, which is negative for output that is too low
        # and positive for output that is too high. For now, the offset is a constant! This can
        # be changed to an offset function
        self.offset = offset
        # Each sensor will be subject to noise, the type of which still has to be decided upon.
        # (Gaussian noise for now) the noise variable (for now) is the standard deviation.
        self.noise_sd = noise_sd

    def noise(self, val):
        return val + np.random.normal(0, self.noise_sd) + self.offset
