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
        self.test_sensors()

    def simulate(self, n_simulations) -> None:
        self.input = self.gen_input(n_simulations)
        print("Input Generated")
        self.output = self.gen_output(self.input)
        print("Output computed & Data Saved")


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


    def gen_input(self, num: int) -> List[Input]:
        input_list = []

        for i in range(num):
            velocity = np.asarray([
                [np.random.rand() * Const.MAX_VELOCITY * (1 if 0 > np.random.rand() >= 0.3 else 0 if 0.3 > np.random.rand() >= 0.6 else -1)],
                [np.random.rand() * Const.MAX_VELOCITY * (1 if 0 > np.random.rand() >= 0.3 else 0 if 0.3 > np.random.rand() >= 0.6 else -1)]
            ]) if np.random.rand() > 0.5 else np.asarray([[0], [0]])
            center = np.asarray([
                [np.random.rand() * self.width],
                [np.random.rand() * self.height]
            ])
            shape = Const.SHAPES[np.random.randint(len(Const.SHAPES))]
            frames = math.ceil(np.random.rand() * Const.MAX_FRAMES)
            force = np.random.rand() * Const.MAX_FORCE

            simulation_class = Class(
                shape_size=Const.BIG(shape),
                movement=abs(np.linalg.norm(velocity)) > 0,  # Velocity vector has a norm higher than 0
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
        version = Const.DATASET_VERSION
        if not os.path.exists(f'../out/v{version}'):
            os.makedirs(f'../out/v{version}')
        with open(f'../out/v{version}/dataset.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "frame", "big/small", "dynamic/static", "press/tap", "dangeours/safe"] +
                ["S" + str(number) for number in range(self.n_sensors)] +
                ["shape", "pressure", "velocity"]
            )

        t1 = time.time()
        output = []
        pool = mp.Pool(mp.cpu_count())
        for i in range(len(inp)):
            output = (pool.apply(self.compute_frames, args=(i, inp[i], version)))
        pool.close()
        t2 = time.time()

        print(f"Simulation used {t2 - t1} seconds")

        self.write_sensor_map_image(version)
        self.write_sensor_adjacency_matrix(version)

        return output

    def compute_frames(self, idn, example, version):
        out = Output(idn, example.shape, example.simulation_class)
        shape = example.shape

        # Compute frames readings
        for _ in range(example.frames):
            example.update_frame(np.array([[0], [0]]).astype(float))
            self.heatmap.nodes = shape.compute_pressure()
            out.reading.append(self.heatmap.get_heatmap_copy())
            temp = self.heatmap.get_heatmap_copy()
            self.heatmap.nodes = temp * 0.8
            example.update_frame()

        # Write frames computed
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

            third_part = [out.shape.shape, out.shape.force, np.linalg.norm(example.vel)]
            with open(f'../out/v{version}/dataset.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(first_part + second_part + third_part)

        # Write padding for remaining empty frames
        for frame in range(Const.MAX_FRAMES - len(out.reading)):
            first_part = [
                str(out.id),
                len(out.reading) + frame,
                int(out.simulation_class.big),
                int(out.simulation_class.moving),
                int(out.simulation_class.press),
                int(out.simulation_class.dangerous)
            ]

            second_part = [0 for _ in self.heatmap.sensors]

            third_part = [out.shape.shape, out.shape.force, np.linalg.norm(example.vel)]
            with open(f'../out/v{version}/dataset.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(first_part + second_part + third_part)

        return out

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

            for i in range(len(self.heatmap.sensors)):
                row = []
                for j in range(len(self.heatmap.sensors)):
                    if i == j:
                        row.append(0)
                    else:
                        dist = math.dist(self.heatmap.sensors[i].center, self.heatmap.sensors[j].center)
                        val = dist if dist < Const.NEAREST_NEIGHBOUR_THRESHOLD else 0
                        row.append(val)
                if len(row) > 0:
                    writer.writerow(row)

    def test_sensors(self):
        for sensor in self.heatmap.sensors:
            prova = np.zeros(shape=self.heatmap.sensors_map.shape)
            for value in sensor.coords:
                    prova[value[0], value[1]] = 1
            break

    def get_sensors_readings(self, frame):
        readings = []
        for sensor in self.heatmap.sensors:
            reading = (sensor.map * frame)
            mask = reading > 0
            if np.sum(reading[mask]) == 0:
                reading = 0
            else:
                reading = np.mean(reading[mask])
            readings.append(reading)
        return readings



if __name__ == "__main__":
    sim = Simulator(
        w=250,
        h=250
    )
    sim.simulate(Const.N_SIMULATIONS)
    # sim.show_readings()
    # sim.create_database()
    # TODO: Optimize code by saving a map for every single sensor at the beginning in order to vectorize operations
