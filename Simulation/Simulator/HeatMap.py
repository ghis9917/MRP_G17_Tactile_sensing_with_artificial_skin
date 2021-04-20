from itertools import product

import numpy as np


class HeatMap:
    def __init__(self, w, h, sensors):
        self.width = w
        self.height = h
        self.nodes: np.ndarray = np.zeros(shape=(w, h))
        self.sensors: np.ndarray = self.create_sensor_map(sensors)

    def create_sensor_map(self, sensors):
        temp = np.zeros(shape=(self.width, self.height))
        for sensor in sensors:
            temp[sensor.x, sensor.y] = 1
        return temp

    def sensor_readings(self):
        return self.sensors * self.nodes

    def get_heatmap_copy(self):
        temp = np.zeros(shape=self.nodes.shape)
        for i, j in product(range(self.width), range(self.height)):
            temp[i, j] = self.nodes[i, j]

        return temp