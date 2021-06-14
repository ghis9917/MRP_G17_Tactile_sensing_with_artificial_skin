# %%
import csv
import itertools
import math
import multiprocessing as mp
import os
import time
from typing import List

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import Simulation.Utils.Constants as Const
from Classes import Class
from Graph import Graph
from HeatMap import HeatMap
from Input import Input
from Output import Output
from Sensors import Sensor
from Shapes import Shape
from Simulation.Simulator.FEM.AbFEM import run_fem
from Simulation.Utils import Utils
from Simulation.Utils.Constants import SMOOTHING, N_CONVOLUTIONS, KERNEL_5_GAUSSIAN
from Simulation.Visualization.Visualizer import Visualizer


class Simulator:
    def __init__(self, w: int, h: int, number_of_sensors):
        self.width = w
        self.height = h
        self.input: List[Shape] = []
        self.output: List[Output] = []

        # self.graph = self.initialize_graph(n_sensors, offset, noise, distribution)
        self.heatmap = HeatMap(w, h, number_of_sensors)
        print("Heatmap Initialized")
        self.n_sensors: int = number_of_sensors

    def simulate(self, n_simulations) -> None:
        self.input = self.gen_input(n_simulations)
        print("Input Generated")
        self.output = self.gen_output(self.input)
        print("Output computed & Data Saved")

    def show_readings(self) -> None:
        counter = 0
        for out in self.output:
            resized_outputs = [cv2.resize(output, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for output in
                               out.reading]
            viz1 = Visualizer(
                cv2.resize(self.heatmap.nodes, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
            viz1.ani_3D(resized_outputs, cv2.resize(self.heatmap.sensors_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
            viz1.ani_2D(resized_outputs, cv2.resize(self.heatmap.sensors_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
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

    def gen_input(self, num: int) -> List[Input]:
        # TODO: test if values correspond to expected class
        input_list = []

        for big, dynamic, press, dangerous in itertools.product(range(2), range(2), range(2), range(2)):
            for _ in range(num):
                if big:
                    shape = Const.BIG_SHAPES[np.random.randint(len(Const.BIG_SHAPES))]
                else:
                    shape = Const.SMALL_SHAPES[np.random.randint(len(Const.SMALL_SHAPES))]

                if dynamic:
                    velocity = np.asarray([[0], [0]])
                else:
                    velocity = np.asarray([
                        [np.random.rand() * Const.MAX_VELOCITY * (
                            1 if np.random.rand() >= 0.5 else -1)],
                        [np.random.rand() * Const.MAX_VELOCITY * (
                            1 if np.random.rand() >= 0.5 else -1)]
                    ])

                if press:
                    frames = np.random.randint(Const.THRESHOLD_PRESS, Const.MAX_FRAMES)
                else:
                    frames = np.random.randint(0, Const.THRESHOLD_PRESS)

                if dangerous:
                    force = np.random.uniform(Const.THRESHOLD_DANGEROUS, Const.MAX_FORCE)
                else:
                    force = np.random.uniform(0, Const.THRESHOLD_DANGEROUS)

                center = np.asarray([
                    [np.random.rand() * self.width],
                    [np.random.rand() * self.height]
                ])

                simulation_class = Class(
                    shape_size=bool(big),
                    movement=bool(dynamic),
                    touch_type=bool(press),
                    dangerous=bool(dangerous)
                )
                input_list.append(Input(
                    shape=Shape(center, force, shape, self.width, self.height),
                    vel=velocity,
                    frames=frames,
                    simulation_class=simulation_class
                ))
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
            output.append(pool.apply(self.compute_frames, args=(i, inp[i], version)))
        pool.close()
        t2 = time.time()

        print(f"Simulation used {t2 - t1} seconds")

        self.write_sensor_map_image(version)
        # self.write_sensor_adjacency_matrix(version)

        return output

    def compute_frames(self, idn, example, version):
        out = Output(idn, example.shape, example.simulation_class)
        shape = example.shape

        print(shape.force)
        print(example.vel)
        print(example.frames)

        # Compute frames readings
        for i in range(Const.MAX_FRAMES):
            if i < example.frames:
                example.update_frame(np.array([[0], [0]]).astype(float))
                displacements = run_fem(shape.current_map, layers=Const.LAYERS, max_force=shape.force, mesh_size=1, vis=False)
                self.heatmap.nodes += displacements
                example.update_frame()

            temp = self.heatmap.get_heatmap_copy()
            out.reading.append(temp)
            self.heatmap.nodes = temp * 0.1

            first_part = [
                str(out.id),
                i,
                int(out.simulation_class.big),
                int(out.simulation_class.moving),
                int(out.simulation_class.press),
                int(out.simulation_class.dangerous)
            ]

            second_part = self.get_sensors_readings(out.reading[i])

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

    def write_sensor_map_image(self, version):
        sensor_placement = Image.fromarray(np.uint8((self.heatmap.sensors_map > 0) * 255), 'L')
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

    def get_sensors_readings(self, frame):
        readings = []
        for x, y in itertools.product(range(self.height), range(self.width)):
            if self.heatmap.sensors_map[x, y]:
                readings.append(frame[x, y])
        return readings


if __name__ == "__main__":
    sim = Simulator(
        w=15,
        h=15,
        number_of_sensors=40
    )
    # sim.simulate(Const.N_SIMULATIONS)
    sim.simulate(1)
    # sim.show_readings()
