from typing import List
import numpy as np
from Classes import Class
from Shapes import Shape


class Output:
    def __init__(self, id: int, shape: Shape, simulation_class: Class):
        self.id = id
        self.reading: List[np.ndarray] = []
        self.shape: Shape = shape
        self.simulation_class: Class = simulation_class