# %%
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations

# import cv2


# Sensor class?
# Grid/ graph class?
# Input generator that interacts with the graph
import numpy as np


class Sensor:
    def __init__(self, x, y, offset, noise_sd):
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
    def __init__(self, sensors):
        self.sensors = sensors
        self.edges = []
        self.faces = []
        self.init_edges_and_faces()

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
