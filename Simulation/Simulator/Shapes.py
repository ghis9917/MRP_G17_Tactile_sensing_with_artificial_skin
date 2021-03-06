import numpy as np
import cv2


class Shape:
    MAP_PATH = '../input/'

    def __init__(self, center: np.ndarray, force: float, map: str, width, height):
        self.force: float = force
        self.center: np.ndarray = center
        self.current_map: np.ndarray = self.load_shape(map, width, height)
        self.traslation: np.ndarray = np.asarray([[0], [0]]).astype(float)
        self.shape = map

    def is_in(self, x, y):
        return self.current_map[x, y] > 0

    def compute_pressure(self):
        return self.force * (self.current_map / 255)

    def update_center(self, vel: np.ndarray) -> None:
        self.center += vel

    def set_traslated_map(self, vector):
        self.traslation += vector
        self.current_map = self.shift(self.traslation)

    # https://stackoverflow.com/questions/54274185/shifting-an-image-by-x-pixels-to-left-while-maintaining-the-original-shape/54274222
    def shift(self, vec):
        num_rows, num_cols = self.img.shape[:2]
        translation_matrix = np.float32([
            [1, 0, vec[0, 0] + self.center[0, 0] - round(self.img.shape[0] / 2)],
            [0, 1, vec[1, 0] + self.center[1, 0] - round(self.img.shape[1] / 2)]
        ])
        return cv2.warpAffine(self.img, translation_matrix, (num_cols, num_rows))

    def load_shape(self, map: str, w, h):
        self.img = cv2.resize(cv2.imread(self.MAP_PATH + map, cv2.IMREAD_GRAYSCALE), dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        return self.img
