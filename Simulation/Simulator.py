# %%
from typing import List
import numpy as np
import Utils
from Constants import SMOOTHING, N_CONVOLUTIONS
from Graph import Graph
from Sensors import Sensor
from Shapes import Ellipse, Shape
from Visualizer import Visualizer


class Simulator:
    def __init__(self, w: int, h: int, n_sensors: int, offset: int, noise: int, distribution: str):
        self.graph = self.initialize_graph(w, h, n_sensors, offset, noise, distribution)
        self.input: List[Shape] = []
        self.output: List = []

    def simulate(self) -> None:
        self.input = self.gen_input(1, 50)
        self.output = self.gen_output(self.input)

    def show_readings(self) -> None:
        for line in self.output:
            viz = Visualizer(self.create_histogram(line[3:-1]), self.graph.sensors, self.input[0])
            viz.plot()

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
                tempx = round(np.random.rand() * width)
                # if np.random.rand() > 0.5:
                #     tempx *= -1
                tempy = round(np.random.rand() * height)
                # if np.random.rand() > 0.5:
                #     tempy *= -1

                temp_offset = np.random.rand() * offset_range
                temp_noise = np.random.rand() * noise_range

                sensors.append(Sensor(i, tempx, tempy, temp_offset, temp_noise))
        return Graph(sensors, width, height)

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
            el_width = np.random.rand() * ellipse_width
            el_height = np.random.rand() * ellipse_height
            input_list.append(Ellipse(temp_x, temp_y, el_width, el_height, temp_f))

        return input_list

    def gen_output(self, inp: List[Shape]) -> List:
        frame = 0
        idn = 0
        output = []
        for ellipse in inp:
            # Output is a list
            # containing "class", "id", "frame", "sensor_1" .... "sensor_n"
            sensor_output = []
            for sensor in self.graph.sensors:
                if ellipse.is_in(sensor.x, sensor.y):
                    sensor_output.append(sensor.noise(ellipse.force))
                    sensor.reading = sensor.noise(ellipse.force)
                else:
                    sensor_output.append(0)
            frame_l = []
            frame_l.append("ellipse")
            frame_l.append(idn)
            frame_l.append(frame)
            for s in sensor_output:
                frame_l.append(s)
            output.append(frame_l)
            frame += 1
            idn += 1

        return output

    def create_histogram(self, l: List) -> np.ndarray:
        hg = np.zeros(shape=(self.graph.width + 1, self.graph.height + 1))
        for i in range(len(l)):
            hg[self.graph.sensors[i].y, self.graph.sensors[i].x] = l[i]
        kernel_3_gaussian = np.array([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]
        ])
        kernel_5_gaussian = 1 / 256 * np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ])
        new = hg
        if SMOOTHING:
            for i in range(N_CONVOLUTIONS):
                new = Utils.convolve2D(new, kernel_5_gaussian, padding=2)
        return new


if __name__ == "__main__":
    sim = Simulator(100, 100, 100, 5, 6, "random")
    sim.simulate()
    sim.show_readings()


