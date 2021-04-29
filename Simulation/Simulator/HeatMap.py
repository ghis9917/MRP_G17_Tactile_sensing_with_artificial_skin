from itertools import product

import numpy as np


class HeatMap:
    def __init__(self, w, h, sensors):
        self.width = w
        self.height = h
        self.nodes: np.ndarray = np.zeros(shape=(w, h))
        self.sensors: np.ndarray = self.create_sensor_map(sensors)

    def create_sensor_map(self, sensors):
        temp = []
        print("test")
        for x in range(self.width):
            templ = []
            for y in range(self.height):
                templ.append(None)
            temp.append(templ)

        for sensor in sensors:
            temp[sensor.x][sensor.y] = sensor

        return temp

    def sensor_readings(self):
        s1 = len(self.sensors)
        s2 = len(self.sensors[0])
        temp = np.zeros(shape=(s1, s2))
        for x, y in product(range(s1), range(s2)):
            if self.sensors[x][y] is not None:
                temp[x][y] = self.sensors[x][y].get_reading(self.nodes[x, y])
                print("sensor reading worked ", temp[x, y])
            else:
                temp[x, y] = 0
        return temp

    def get_heatmap_copy(self):
        temp = np.zeros(shape=self.nodes.shape)
        for i, j in product(range(self.width), range(self.height)):
            temp[i, j] = self.nodes[i, j]

        return temp
