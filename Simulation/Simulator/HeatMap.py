from itertools import product
from typing import List

import numpy as np
import PIL.Image as Image

from Sensors import Sensor
import Utils.Constants as Const


class HeatMap:
    def __init__(self, w, h, sensors_map):
        self.width = w
        self.height = h
        self.nodes: np.ndarray = np.zeros(shape=(w, h))
        self.sensors_map: np.ndarray = sensors_map#self.create_sensor_map(sensors_map)
        self.sensors: List[Sensor] = self.build_sensors_list()

    def create_sensor_map(self, sensors):
        temp = np.zeros(shape=(self.width, self.height))
        for sensor in sensors:
            temp[sensor.x, sensor.y] = 1
        return temp

    def check_what_this_is(self, sensors):
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

    # Recursive version of the fill flood algorithm
    # def fill_flood(self, row, col, current_single_sensor_list):
    #     for (i, j) in [(0, -1), (-1, 0), (+1, 0), (0, +1)]:
    #         try:
    #             if self.sensors_map[row + i, col + j] == 1 and not (row + i, col + j) in current_single_sensor_list:
    #                 current_single_sensor_list.append((row + i, col + j))
    #                 current_single_sensor_list = self.recursive_neighbours_call(row + i, col + j, current_single_sensor_list)
    #         except Exception as e:
    #             print(e)
    #             continue
    #     return current_single_sensor_list

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

