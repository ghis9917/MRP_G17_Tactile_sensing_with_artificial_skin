from typing import List

import numpy as np


class Output:
    def __init__(self, id: int):
        self.id = id
        self.reading: List[np.ndarray] = []