import math
import numpy as np
from Utils import Utils
from Utils.Constants import DISTANCE_WEIGHT


class Shape:
    def __init__(self, center, force):
        self.force = force
        self.center: np.ndarray = center

    def is_in(self, x, y):
        pass

    def compute_pressure(self, x, y):
        pass

    def update_center(self, vel:np.ndarray) -> None:
        self.center += vel


class Ellipse(Shape):
    # ellipse formula: (x-h)**2/a**2 + (y-k)**2/b**2 = 1
    # h, k -> center
    # a, b -> width, height

    def __init__(self, center, force, h, k, a, b):
        super().__init__(center, force)
        self.h = h
        self.k = k
        self.a = a
        self.b = b

    def is_in(self, x, y):
        val = (x - self.center[0]) ** 2 / self.a ** 2 + (y - self.center[1]) ** 2 / self.b ** 2
        if val <= 1:
            return True
        else:
            return False

    def compute_pressure(self, x, y):
        if self.is_in(x, y):
            return self.force
        else:
            return 0
            # point_on_ellipse = Utils.distance_ellipse_point(self.a, self.b, (x - self.center[0], y - self.center[1]))
            # point_on_ellipse = (point_on_ellipse[0] + self.center[0], point_on_ellipse[1] + self.center[1])
            # return self.gaussian_pressure(point_on_ellipse[0], point_on_ellipse[1], x, y, self.force)

    @staticmethod
    def gaussian_pressure(x1, y1, x2, y2, force):
        distance = math.dist((x1, y1), (x2, y2))
        return force * (np.exp(-(distance * DISTANCE_WEIGHT) ** 2))
