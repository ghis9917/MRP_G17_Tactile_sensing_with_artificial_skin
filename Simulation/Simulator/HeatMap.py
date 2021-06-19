import itertools
from itertools import product
from typing import List

import numpy as np
import PIL.Image as Image

from Sensors import Sensor
import Simulation.Utils.Constants as Const


class HeatMap:
    def __init__(self, w, h, number_of_sensors):
        cm_2_in = 0.3937 # Instantiate the model
        self.width = round(20 * cm_2_in)
        self.height = round(20 * cm_2_in)
        self.nodes: np.ndarray = np.zeros(shape=(self.width, self.height))
        self.sensors_map = self.create_sensor_map()
        # self.sensors_map: np.ndarray = sensors_map#self.create_sensor_map(sensors_map)
        # self.sensors: List[Sensor] = self.build_sensors_list()

    def create_sensor_map(self, ):
        return np.ones(shape=(self.width, self.height))

    def check_what_this_is(self, sensors):
        temp = []
        for x in range(self.width):
            templ = []
            for y in range(self.height):
                templ.append(None)
            temp.append(templ)

        for sensor in sensors:
            temp[sensor.x][sensor.y] = sensor

        return temp

    def sensor_readings(self):
        return self.sensors_map * self.nodes
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

    def build_sensors_list(self):
        sensors_list = []
        picked_already = []
        for row, col in product(range(self.sensors_map.shape[1]), range(self.sensors_map.shape[0])):
            temp = []
            if self.sensors_map[row, col] == 1 and not (row, col) in picked_already:
                all_temp = self.fill_flood(row, col)
                picked_already += all_temp
                temp += all_temp
                sensors_list.append(temp)
        final = []
        counter = 0
        for sensor in sensors_list:
            xs = [coord[0] for coord in sensor]
            ys = [coord[1] for coord in sensor]
            final.append(Sensor(
                counter,
                xs,
                ys,
                sensor,
                np.random.rand() * Const.MAX_OFFSET,
                np.random.rand() * Const.MAX_NOISE,
                self.sensors_map.shape
            ))
            counter += 1
        return final

    # Fill flood algorithm with queue, avoids stack overflow for recusive calls
    def fill_flood(self, row, col):
        Q = [(row, col)]
        S = []
        while len(Q) > 0:
            n = Q.pop(0)
            try:
                if self.sensors_map[n[0], n[1]] == 1 and not (n[0], n[1]) in S:
                    S.append((n[0], n[1]))
                    Q.append((n[0] - 1, n[1]))
                    Q.append((n[0] + 1, n[1]))
                    Q.append((n[0], n[1] - 1))
                    Q.append((n[0], n[1] + 1))
            except Exception as e:
                print(e)
                continue
        return S

