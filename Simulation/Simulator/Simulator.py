# %%
import math
from itertools import product
from typing import List
import numpy as np

from Output import Output
from Utils import Utils
from Utils.Constants import SMOOTHING, N_CONVOLUTIONS, KERNEL_5_GAUSSIAN
from Graph import Graph
from HeatMap import HeatMap
from Sensors import Sensor
from Shapes import Shape
from Visualization.Visualizer import Visualizer
from Input import Input


class Simulator:
    def __init__(self, w: int, h: int, n_sensors: int, offset: int, noise: int, distribution: str):
        self.graph, self.heatmap = self.initialize_graph(w, h, n_sensors, offset, noise, distribution)
        self.input: List[Shape] = []
        self.output: List = []

    def simulate(self) -> None:
        self.input = self.gen_input(1, 50)
        self.output = self.gen_output(self.input)

    def show_readings(self) -> None:
        for out in self.output:
            viz1 = Visualizer(self.heatmap.nodes, self.heatmap.sensor_readings(), self.input[0].shape)
            # viz1.ani_3D(out.reading, self.heatmap.sensors)
            viz1.ani_2D(out.reading, self.heatmap.sensors)

    # Width & Height of sheet of skin
    # Number of Sensors
    # Range x ([-x, x]) from which an offset is drawn at random
    # Range x ([-x, x]) from which a noise is drawn at random
    # Type of sensor distribution (TODO Implement different sensor distributions)
    @staticmethod
    def initialize_graph(width: int, height: int, num_sensors: int, offset_range: int, noise_range: int, distribution_type: str ="random") -> Graph:
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
    def gen_input(self, num: int, force_range: float) -> List[Shape]:
        input_list = []
        ellipse_width, ellipse_height = 40, 40

        for i in range(num):
            temp_f = np.random.rand() * force_range
            temp_x = np.random.rand() * self.graph.width
            temp_y = np.random.rand() * self.graph.height
            velocity = np.asarray([[1], [1]])
            center = np.asarray([[temp_x], [temp_y]])
            print(center)
            input_list.append(Input(Shape(center, temp_f, 'hand.png'), velocity, 30))

        return input_list

    def gen_output(self, inp: List[Input]) -> List:
        idn = 0
        output = []
        for example in inp:
            out = Output(idn)
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


if __name__ == "__main__":
    sim = Simulator(100, 100, 40, 5, 6, "random")
    sim.simulate()
    sim.show_readings()


