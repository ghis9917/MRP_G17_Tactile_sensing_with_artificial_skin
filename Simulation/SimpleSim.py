# %%
from typing import List

import numpy as np
import scipy
from scipy.ndimage import convolve
from scipy.spatial import Delaunay
from itertools import combinations

import Utils
from Shapes import Ellipse
from Visualizer import Visualizer
# import cv2


# Sensor class?
# Grid/ graph class?
# Input generator that interacts with the graph
import numpy as np

global graph


class Sensor:
    def __init__(self, id_num, x, y, offset, noise_sd):
        self.id = id_num
        self.x = x
        self.y = y
        # Each sensor will have a certain offset, which is negative for output that is too low
        # and positive for output that is too high. For now, the offset is a constant! This can
        # be changed to an offset function
        self.offset = offset
        # Each sensor will be subject to noise, the type of which still has to be decided upon.
        # (Gaussian noise for now) the noise variable (for now) is the standard deviation.
        self.noise_sd = noise_sd

    def noise(self, val):
        return val + np.random.normal(0, self.noise_sd) + self.offset


class Graph:
    def __init__(self, sensors, width, height):
        self.sensors = sensors
        self.edges = []
        self.faces = []
        self.init_edges_and_faces()
        self.width = width
        self.height = height

    def init_edges_and_faces(self):
        # TODO triangulation
        # Calculate Delaunay triangulation to get edges and faces of mesh
        vertices = []
        for v in self.sensors:
            vertices.append([v.x, v.y])
        vertices_for_delaunay = np.array(vertices)
        tri = Delaunay(vertices_for_delaunay)
        for triangle in tri.simplices:
            self.edges.extend(set(combinations(triangle, 2)))
            self.faces.append(triangle)


class Output:
    def __init__(self, x, y, reading):
        self.x = x
        self.y = y
        self.reading = reading


# Width & Height of sheet of skin
# Number of Sensors
# Range x ([-x, x]) from which an offset is drawn at random
# Range x ([-x, x]) from which a noise is drawn at random
# Type of sensor distribution (TODO For now, only random is implemented)
def initialize_graph(width, height, num_sensors, offset_range, noise_range, distribution_type="random"):
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
    global graph
    graph = Graph(sensors, width, height)


# Number of input
# Range x ([0, x]) from which the amount of force of every input is drawn
# TODO for now input is based on ellipses. Try image based approach for a wider range of shapes
def gen_input(num, force_range):
    input_list = []
    ellipse_width, ellipse_height = 40, 40
    for i in range(num):
        temp_f = np.random.rand() * force_range
        temp_x = np.random.rand() * graph.width
        temp_y = np.random.rand() * graph.height
        el_width = np.random.rand() * ellipse_width
        el_height = np.random.rand() * ellipse_height
        input_list.append(Ellipse(temp_x, temp_y, el_width, el_height, temp_f))

    return input_list


def gen_output(inp):
    frame = 0
    idn = 0
    output = []
    for ellipse in inp:
        # Output is a list
        # containing "class", "id", "frame", "sensor_1" .... "sensor_n"
        sensor_output = []
        for sensor in graph.sensors:
            if ellipse.is_in(sensor.x, sensor.y):
                sensor_output.append(sensor.noise(ellipse.force))
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


initialize_graph(100, 100, 100, 5, 6)
out = gen_output(gen_input(1, 50))


def create_histogram(g: Graph, l: List):
    hg = np.zeros(shape=(g.width+1, g.height+1))
    for i in range(len(l)):
        hg[g.sensors[i].x, g.sensors[i].y] = l[i]
    kernel_3_gaussian = np.array([
        [1 / 16, 1 / 8, 1 / 16],
        [1 / 8, 1 / 4, 1 / 8],
        [1 / 16, 1 / 8, 1 / 16]
    ])
    kernel_5_gaussian = 1/256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    new = hg
    for i in range(50):
        new = Utils.convolve2D(new, kernel_5_gaussian, padding=2)
    return new



for line in out:
    histogram = create_histogram(graph, line[3:-1])
    viz = Visualizer(histogram)
    viz.plot()
    print(line)
