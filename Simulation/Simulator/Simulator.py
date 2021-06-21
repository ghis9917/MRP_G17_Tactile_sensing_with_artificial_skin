# %%
import contextlib
import csv
import itertools
import math
import multiprocessing as mp
import os
import threading
import time
from typing import List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import pandas as pd
import Simulation.Utils.Constants as Const
from Classes import Class
from Graph import Graph
from HeatMap import HeatMap
from Input import Input
from Output import Output
from Sensors import Sensor
from Shapes import Shape
from Simulation.Simulator.FEM.AbFEM import run_fem
from Simulation.Visualization.Visualizer import Visualizer

datalock = threading.RLock()

class Simulator:
    def __init__(self, w: int, h: int, number_of_sensors):
        self.width = w
        self.height = h
        self.input: List[Shape] = []
        self.output: List[Output] = []

        self.heatmap = HeatMap(w, h, number_of_sensors)
        print("Heatmap Initialized")
        self.n_sensors: int = number_of_sensors
        self.out_dataframe = pd.DataFrame(columns=["version",
                                                   "idn",
                                                   "i",
                                                   "displacements_surface",
                                                   "displacements",
                                                   "displacements_under",
                                                   "force_at_surface_matrix"])

    def simulate(self, n_simulations) -> None:
        self.input = self.gen_input(n_simulations)
        print("Input Generated")
        self.gen_output(self.input)
        # self.save_dataframe()
        print("Output computed & Data Saved")

    def show_readings(self) -> None:
        counter = 0
        for out in self.output:
            resized_outputs = [cv2.resize(output, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for output in
                               out.reading]
            viz1 = Visualizer(
                cv2.resize(self.heatmap.nodes, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
            viz1.ani_3D(resized_outputs,
                        cv2.resize(self.heatmap.sensors_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
            viz1.ani_2D(resized_outputs,
                        cv2.resize(self.heatmap.sensors_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
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
        input_list = []

        for big, dynamic, press, dangerous in itertools.product(range(2), range(2), range(2), range(2)):
            for _ in range(num):
                if big:
                    if dynamic:
                        shape = Const.BIG_SHAPES[np.random.randint(len(Const.BIG_SHAPES))]
                    else:
                        shape = (Const.BIG_SHAPES + Const.BIG_STATIC_SHAPES)[
                            np.random.randint(len(Const.BIG_SHAPES) + len(Const.BIG_STATIC_SHAPES))]
                else:
                    if dynamic:
                        shape = Const.SMALL_SHAPES[np.random.randint(len(Const.SMALL_SHAPES))]
                    else:
                        shape = (Const.SMALL_SHAPES + Const.SMALL_STATIC_SHAPES)[
                            np.random.randint(len(Const.SMALL_SHAPES) + len(Const.SMALL_STATIC_SHAPES))]

                if dynamic:
                    velocity = np.asarray([
                        [np.random.rand() * Const.MAX_VELOCITY * (
                            1 if np.random.rand() >= 0.5 else -1)],
                        [np.random.rand() * Const.MAX_VELOCITY * (
                            1 if np.random.rand() >= 0.5 else -1)]
                    ])
                else:
                    velocity = np.asarray([[0], [0]])

                if press:
                    frames = np.random.randint(Const.THRESHOLD_PRESS, Const.MAX_FRAMES)
                else:
                    frames = np.random.randint(0, Const.THRESHOLD_PRESS)

                N_t = self.Pa_to_N(shape)
                if dangerous:
                    force = np.random.uniform(N_t, Const.MAX_FORCE)
                else:
                    force = np.random.uniform(0, N_t)

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

    def gen_output(self, inp: List[Input]) -> None:
        version = Const.DATASET_VERSION
        if not os.path.exists(f'../out/v{version}'):
            os.makedirs(f'../out/v{version}')
        with open(f'../out/v{version}/dataset.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "frame", "big/small", "dynamic/static", "press/tap", "dangeours/safe"] +
                ["S" + str(number) for number in range(len(self.heatmap.sensors_map.flatten()))] +
                ["shape", "pressure", "velocity"]
            )

        with open(f'../out/v{version}/v_{version}_smoothout.csv', "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['version', 'id', 'frame', 'displacements_surface', 'displacements',
                 'displacements_under', 'force_at_surface_matrix'] +
                ["big/small", "dynamic/static", "press/tap", "dangeours/safe"] +
                ["shape", "pressure", "velocity"]
            )

        t1 = time.time()
        pool = mp.Pool(mp.cpu_count())

        for i in range(len(inp)):
            pool.apply_async(self.compute_frames,
                             args=(i, inp[i], version))

        pool.close()
        pool.join()

        # new_df = pd.DataFrame(data_row)
        # self.out_dataframe = pd.concat([self.out_dataframe, data_row], axis=0, ignore_index=True)

        t2 = time.time()

        print(f"Simulation used {t2 - t1} seconds")

        self.write_sensor_map_image(version)
        self.write_sensor_adjacency_matrix(version)

    def compute_frames(self, idn, example, version):
        out = Output(idn, example.shape, example.simulation_class)
        shape = example.shape

        def log_results(result):
            with open(f'../out/v{version}/v_{version}_smoothout.csv', "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
                f.close()

        # Compute frames readings
        for i in tqdm(range(Const.MAX_FRAMES)):
            # displacements_surface, displacements = pd.nan, pd.nan
            # displacements_under, force_at_surface_matrix = pd.nan, pd.nan

            if i < example.frames:
                example.update_frame(np.array([[0], [0]]).astype(float))
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # Ignore prints in function
                    displacements_surface, displacements, displacements_under, force_at_surface_matrix = run_fem(
                        shape.current_map,
                        layers=Const.LAYERS,
                        max_force=shape.force,
                        mesh_size=1,
                        vis=False
                    )

                    log_results(
                        [
                            version,
                            str(out.id),
                            i,
                            displacements_surface,
                            displacements,
                            displacements_under,
                            force_at_surface_matrix,
                            int(out.simulation_class.big),
                            int(out.simulation_class.moving),
                            int(out.simulation_class.press),
                            int(out.simulation_class.dangerous),
                            out.shape.shape,
                            out.shape.force,
                            np.linalg.norm(example.vel)
                        ]
                    )

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

            # sensor_values = out.reading[i]*self.heatmap.sensors_map + self.heatmap.sensors_map
            # second_part = list(sensor_values[np.nonzero(sensor_values)] - 1)
            second_part = self.get_sensors_readings(out.reading[i])

            third_part = [out.shape.shape, out.shape.force, np.linalg.norm(example.vel)]

            with open(f'../out/v{version}/dataset.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(first_part + second_part + third_part)

        self.output.append(out)

    def write_sensor_map_image(self, version):
        sensor_placement = Image.fromarray(np.uint8((self.heatmap.sensors_map > 0) * 255), 'L')
        sensor_placement.save(f'../out/v{version}/sensor_map.png')

    def write_sensor_adjacency_matrix(self, version):
        with open(f'../out/v{version}/adjacency_matrix.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["S" + str(number) for number in range(len(self.heatmap.sensors_map.flatten()))])

            for i1, j1 in itertools.product(range(self.heatmap.width), range(self.heatmap.height)):
                row = []
                for i2, j2 in itertools.product(range(self.heatmap.width), range(self.heatmap.height)):
                    if i1 != i2 or j1 != j2:
                        dist = math.dist((i1, j1), (i2, j2))
                        if dist <= math.sqrt(2)+0.1:
                            row.append(dist)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                if len(row) > 0:
                    writer.writerow(row)

    def get_sensors_readings(self, frame):
        readings = []
        for x, y in itertools.product(range(self.heatmap.height), range(self.heatmap.width)):
            if self.heatmap.sensors_map[x, y]:
                readings.append(frame[x, y])
        return readings

    def Pa_to_N(self, shape):
        img = cv2.resize(
            cv2.imread('../input/' + shape, cv2.IMREAD_GRAYSCALE),
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_CUBIC) / 255
        img = img / np.sum(img)
        newton_threshold = Const.THRESHOLD_DANGEROUS * ((Const.SIZE / self.width) ** 2) / np.max(img)
        return newton_threshold

    def save_dataframe(self):
        version = Const.DATASET_VERSION
        self.out_dataframe.to_csv(f'../out/v{version}/v_{version}_moadf.csv', index=False)
        np.save(f'../out/v{version}/v_{version}.npy', self.out_dataframe, allow_pickle=True)


if __name__ == "__main__":
    sim = Simulator(
        w=20,
        h=20,
        number_of_sensors=40
    )
    sim.simulate(1)
    # sim.show_readings()
