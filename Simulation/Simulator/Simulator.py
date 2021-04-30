# %%

import math
import os
from itertools import product
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm

import Simulation.Utils.Constants as Const

from Output import Output
from Classes import Class
from Simulation.Utils import Utils
from Simulation.Utils.Constants import SMOOTHING, N_CONVOLUTIONS, KERNEL_5_GAUSSIAN
from Graph import Graph
from HeatMap import HeatMap
from Sensors import Sensor
from Shapes import Shape
from Simulation.Visualization.Visualizer import Visualizer
from Input import Input


class Simulator:
    def __init__(self, w: int, h: int, n_sensors: int, offset: int, noise: int, distribution: str, shape: str):
        self.graph, self.heatmap = self.initialize_graph(w, h, n_sensors, offset, noise, distribution)
        self.input: List[Shape] = []
        self.output: List[Output] = []
        self.n_sensors: int = n_sensors

    def simulate(self, n_simulations) -> None:
        self.input = self.gen_input(n_simulations, 50)
        self.output = self.gen_output(self.input)

    def show_readings(self) -> None:
        counter = 0
        for out in self.output:
            viz1 = Visualizer(self.heatmap.nodes, self.heatmap.sensor_readings(), self.input[counter].shape)
            # viz1.ani_3D(out.reading, self.heatmap.sensors)
            viz1.ani_2D(out.reading, self.heatmap.sensors)
            counter += 1

    # Width & Height of sheet of skin
    # Number of Sensors
    # Range x ([-x, x]) from which an offset is drawn at random
    # Range x ([-x, x]) from which a noise is drawn at random
    # Type of sensor distribution (TODO Implement different sensor distributions)
    @staticmethod
    def initialize_graph(width: int, height: int, num_sensors: int, offset_range: int, noise_range: int,
                         distribution_type: str = "random") -> (Graph, HeatMap):
        sensors = []
        if distribution_type == "random":
            for i in range(num_sensors):
                tempx = math.floor(np.random.rand() * width)
                tempy = math.floor(np.random.rand() * height)
                temp_offset = np.random.rand() * offset_range
                temp_noise = np.random.rand() * noise_range
                sensors.append(Sensor(i, tempx, tempy, temp_offset, temp_noise))
        return Graph(sensors, width, height), HeatMap(width, height, sensors)

    # Number of input
    # Range x ([0, x]) from which the amount of force of every input is drawn
    # TODO for now input is based on ellipses. Try image based approach for a wider range of shapes
    def gen_input(self, num: int, force_range: float) -> List[Input]:
        input_list = []
        ellipse_width, ellipse_height = 40, 40

        for i in range(num):
            velocity = np.asarray([
                [np.random.rand() * Const.MAX_VELOCITY],
                [np.random.rand() * Const.MAX_VELOCITY]
            ])
            center = np.asarray([
                [np.random.rand() * self.graph.width],
                [np.random.rand() * self.graph.height]
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
                    shape=Shape(center, force, shape),
                    vel=velocity,
                    frames=frames,
                    simulation_class=simulation_class
                )
            )

        return input_list

    def gen_output(self, inp: List[Input]) -> List:
        idn = 0
        output = []
        for example in tqdm(inp):
            out = Output(idn, example.shape, example.simulation_class)
            shape = example.shape
            for frame in range(example.frames):
                # Output is a list
                # containing "class", "id", "frame", "sensor_1" .... "sensor_n"
                # sensor_output = []
                # for sensor in self.graph.sensors:
                #     if shape.is_in(sensor.x, sensor.y):
                #         sensor_output.append(sensor.noise(shape.force))
                #         sensor.reading = sensor.noise(shape.force)
                #     else:
                #         sensor_output.append(0)

                # HEATMAP UPDATES
                example.update_frame(np.asarray([[0], [0]]).astype(float))
                for i, j in product(range(self.heatmap.width), range(self.heatmap.height)):
                    self.heatmap.nodes[i, j] = shape.compute_pressure(i, j)
                out.reading.append(self.heatmap.get_heatmap_copy())
                temp = self.heatmap.get_heatmap_copy()
                self.heatmap.nodes = temp * 0.8

                example.update_frame()
            output.append(out)
            idn += 1
        return output

    def create_histogram(self, l: List) -> np.ndarray:
        hg = np.zeros(shape=(self.graph.width + 1, self.graph.height + 1))
        for i in range(len(l)):
            hg[self.graph.sensors[i].y, self.graph.sensors[i].x] = l[i]
        if SMOOTHING:
            for i in range(N_CONVOLUTIONS):
                hg = Utils.convolve2D(hg, KERNEL_5_GAUSSIAN, padding=2)
        return hg

    def create_database(self):

        version = 1
        if not os.path.exists(f'../out/v{version}'):
            os.makedirs(f'../out/v{version}')

        # import csv
        # with open(f'../out/v{version}/dataset.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(
        #         ["id", "frame", "big/small", "dynamic/static", "press/tap", "dangeours/safe"] +
        #         ["S" + str(number) for number in range(self.n_sensors)] +
        #         ["shape", "pressure"]
        #     )
        #     for out in self.output:
        #         for frame in range(len(out.reading)):
        #             first_part = [
        #                 str(out.id),
        #                 frame,
        #                 int(out.simulation_class.big),
        #                 int(out.simulation_class.moving),
        #                 int(out.simulation_class.press),
        #                 int(out.simulation_class.dangerous)
        #             ]
        #
        #             current_reading = out.reading[frame]
        #             current_sensor_reading = (current_reading + self.heatmap.sensors) * self.heatmap.sensors
        #
        #             sensors_resing_list = list(current_sensor_reading.flatten())
        #             second_part = [value - 1 for value in sensors_resing_list if value != 0]
        #
        #             third_part = [out.shape.shape, out.shape.force]
        #
        #             writer.writerow(first_part + second_part + third_part)

        sensor_placement = Image.fromarray(np.uint8(self.heatmap.sensors), 'L')
        sensor_placement.save(f'../out/v{version}/sensor_map.png')


if __name__ == "__main__":
    sim = Simulator(
        w=100,
        h=100,
        n_sensors=40,
        offset=5,
        noise=6,
        distribution="random",
        shape='hand.png'
    )
    # sim.simulate(Const.N_SIMULATIONS)
    sim.simulate(1)
    sim.show_readings()
    # sim.create_database()
