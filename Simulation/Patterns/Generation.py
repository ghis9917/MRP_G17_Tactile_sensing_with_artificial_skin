from itertools import product

import cv2
import numpy as np
from PIL import Image


def create_sensors_map(w, h, w_d, h_d, w_offset, h_offset):
    map = np.asarray([255 if (i - w_offset) % w_d == 0 and (j - h_offset) % h_d == 0 else 0 for j, i in product(range(h), range(w))]).reshape(h, w)
    img = Image.fromarray(np.uint8(map), 'L')
    img.save('temp/temp.png')


def create_patter():

    kernel = np.asarray([
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    ], dtype=np.uint8)

    sensors_map = cv2.imread('temp/temp.png', cv2.IMREAD_GRAYSCALE)
    new = cv2.resize(sensors_map, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    new = cv2.dilate(new, kernel, iterations=2)
    new = (1*(new>100)).astype(float)
    cv2.imwrite('out/Pattern40.png', new * 255)


if __name__ == "__main__":
    create_sensors_map(50, 50, round(50/8), round(50/5), 4, 4)
    create_patter()
