# %%
import csv
import itertools
import math
import os
import time
from itertools import product
import multiprocessing as mp
from typing import List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import Utils.Constants as Const

from Output import Output
from Classes import Class
from Utils import Utils
from Utils.Constants import SMOOTHING, N_CONVOLUTIONS, KERNEL_5_GAUSSIAN
from Graph import Graph
from HeatMap import HeatMap
from Sensors import Sensor
from Shapes import Shape
from Visualization.Visualizer import Visualizer
from Input import Input


class Simulator:
    def __init__(self, w: int, h: int):
        self.width = w
        self.height = h
        self.input: List[Shape] = []
        self.output: List[Output] = []

        # self.graph = self.initialize_graph(n_sensors, offset, noise, distribution)
        self.heatmap = self.initialize_heatmap(w, h)
        print("Heatmap Initialized")
        self.n_sensors: int = len(self.heatmap.sensors)

    def simulate(self, n_simulations) -> None:
        self.input = self.gen_input(n_simulations)
        print("Input Generated")
        self.output = self.gen_output(self.input)
        print("Output computed")
        self.create_database()
        print("Data Saved")

    def show_readings(self) -> None:
        counter = 0
        for out in self.output:
            viz1 = Visualizer(self.heatmap.nodes, self.heatmap.sensor_readings(), self.input[counter].shape)
            viz1.ani_3D(out.reading, self.heatmap.sensors_map)
            viz1.ani_2D(out.reading, self.heatmap.sensors_map)
            counter += 1

    def initialize_graph(self, num_sensors: int, offset_range: int, noise_range: int,
                         distribution_type: str = "random") -> (Graph, HeatMap):
        sensors = []
        if distribution_type == "random":
            for i in range(num_sensors):
                tempx = math.floor(np.random.rand() * self.width)
                tempy = math.floor(np.random.rand() * self.height)
                temp_offset = np.random.rand() * offset_range
                temp_noise = np.random.rand() * noise_range
                sensors.append(Sensor(i, tempx, tempy, temp_offset, temp_noise))
        return Graph(sensors, self.width, self.height)

    def initialize_heatmap(self, w, h):
        sensors_map = cv2.resize(
            cv2.imread('../Patterns/out/Pattern40.png', cv2.IMREAD_GRAYSCALE),
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_CUBIC
        )

        sensors_map = 1 * (sensors_map > 0)
        return HeatMap(w, h, sensors_map)

    # Number of input
    # Range x ([0, x]) from which the amount of force of every input is drawn
    # TODO for now input is based on ellipses. Try image based approach for a wider range of shapes
    def gen_input(self, num: int) -> List[Input]:
        input_list = []
        ellipse_width, ellipse_height = 40, 40

        for i in range(num):
            velocity = np.asarray([
                [np.random.rand() * Const.MAX_VELOCITY],
                [np.random.rand() * Const.MAX_VELOCITY]
            ])
            center = np.asarray([
                [np.random.rand() * self.width],
                [np.random.rand() * self.height]
            ])
            shape = Const.SHAPES[np.random.randint(len(Const.SHAPES))]
            frames = math.ceil(np.random.rand() * Const.MAX_FRAMES)
            force = np.random.rand() * Const.MAX_FORCE

            simulation_class = Class(
                shape_size=Const.BIG(shape),
                movement=np.linalg.norm(velocity) > 0,  # Velocity vector has a norm higher than 0
                touch_type=frames > Const.THRESHOLD_PRESS,
                # If the interaction lasts for more than 3 frames than the touch becomes a "press"
                dangerous=force > Const.THRESHOLD_DANGEROUS  # Unit is kPa, higher than 90.5 is considered dangerous
            )

            input_list.append(
                Input(
                    shape=Shape(center, force, shape, self.width, self.height),
                    vel=velocity,
                    frames=frames,
                    simulation_class=simulation_class
                )
            )

        return input_list

    def gen_output(self, inp: List[Input]) -> List:
        version = 3
        if not os.path.exists(f'../out/v{version}'):
            os.makedirs(f'../out/v{version}')
        with open(f'../out/v{version}/dataset.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "frame", "big/small", "dynamic/static", "press/tap", "dangeours/safe"] +
                ["S" + str(number) for number in range(self.n_sensors)] +
                ["shape", "pressure"]
            )

        t1 = time.time()
        pool = mp.Pool(mp.cpu_count())
        for i in range(len(inp)):
            output = pool.apply(self.compute_frames, args=(i, inp[i], version))
        pool.close()
        t2 = time.time()

        print(f"Simulation used {t2 - t1} seconds")

        return output

    def compute_frames(self, idn, example, version):
        out = Output(idn, example.shape, example.simulation_class)
        shape = example.shape
        for _ in range(example.frames):
            example.update_frame(np.asarray([[0], [0]]).astype(float))
            for i, j in product(range(self.heatmap.width), range(self.heatmap.height)):
                self.heatmap.nodes[i, j] = shape.compute_pressure(i, j)
            out.reading.append(self.heatmap.get_heatmap_copy())
            temp = self.heatmap.get_heatmap_copy()
            self.heatmap.nodes = temp * 0.8
            example.update_frame()

        for frame in range(len(out.reading)):
            first_part = [
                str(out.id),
                frame,
                int(out.simulation_class.big),
                int(out.simulation_class.moving),
                int(out.simulation_class.press),
                int(out.simulation_class.dangerous)
            ]

            second_part = self.get_sensors_readings(out.reading[frame])

            third_part = [out.shape.shape, out.shape.force]
            with open(f'../out/v{version}/dataset.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(first_part + second_part + third_part)

    def create_histogram(self, l: List) -> np.ndarray:
        hg = np.zeros(shape=(self.graph.width + 1, self.graph.height + 1))
        for i in range(len(l)):
            hg[self.graph.sensors[i].y, self.graph.sensors[i].x] = l[i]
        if SMOOTHING:
            for i in range(N_CONVOLUTIONS):
                hg = Utils.convolve2D(hg, KERNEL_5_GAUSSIAN, padding=2)
        return hg

    def create_database(self):
        version = 3
        if not os.path.exists(f'../out/v{version}'):
            os.makedirs(f'../out/v{version}')

        self.write_sensor_map_image(version)
        self.write_sensor_adjacency_matrix(version)

    def write_sensor_map_image(self, version):
        sensor_placement = Image.fromarray(np.uint8(self.heatmap.sensors_map * 255), 'L')
        sensor_placement.save(f'../out/v{version}/sensor_map.png')

    def write_sensor_adjacency_matrix(self, version):
        with open(f'../out/v{version}/adjacency_matrix.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["S" + str(number) for number in range(self.n_sensors)])

            for i1, j1 in product(range(self.heatmap.sensors_map.shape[0]), range(self.heatmap.sensors_map.shape[1])):
                row = []
                if self.heatmap.sensors_map[i1, j1] == 1:
                    for i2, j2 in product(range(self.heatmap.sensors_map.shape[0]),
                                          range(self.heatmap.sensors_map.shape[1])):
                        if self.heatmap.sensors_map[i2, j2] == 1:
                            if i2 == i1 and j2 == j1:
                                row.append(0)
                            else:
                                dist = math.dist((i1, j1), (i2, j2))
                                val = dist if dist < Const.NEAREST_NEIGHBOUR_THRESHOLD else 0
                                row.append(val)
                if len(row) > 0:
                    writer.writerow(row)

    def test_sensors(self):
        test_sensor = self.heatmap.sensors[0]
        test_sensor_map = np.zeros(shape=(self.width, self.height))
        for x, y in product(test_sensor.x, test_sensor.y):
            test_sensor_map[x, y] = 1
        sensor_placement = Image.fromarray(np.uint8(test_sensor_map * 255), 'L')
        sensor_placement.show('')

    def get_sensors_readings(self, frame):
        readings = []
        for sensor in self.heatmap.sensors:
            tot = 0
            for x, y in product(sensor.x, sensor.y):
                tot += frame[x, y]
            readings.append(tot/len(sensor.x))
        return readings



if __name__ == "__main__":
    sim = Simulator(
        w=1000,
        h=1000
    )
    # sim.simulate(Const.N_SIMULATIONS)
    sim.simulate(100)
    # sim.show_readings()
    sim.create_database()
    # TODO: test method to detect sensors as group of neighbour pixels
