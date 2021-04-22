from typing import List
import numpy as np
from Classes import Class


class Output:
    def __init__(self, id: int, simulation_class: Class):
        self.id = id
        self.reading: List[np.ndarray] = []
        self.simulation_class: Class = simulation_class