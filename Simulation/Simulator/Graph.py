from itertools import combinations
from typing import List

from scipy.spatial import Delaunay
import numpy as np

from Sensors import Sensor


class Graph:
    def __init__(self, sensors: List[Sensor], width: int, height: int):
        self.sensors = sensors
        self.edges = []
        self.faces = []
        self.init_edges_and_faces()
        self.width = width
        self.height = height

    def init_edges_and_faces(self) -> None:
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