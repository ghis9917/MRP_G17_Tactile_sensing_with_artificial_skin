from typing import List

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, heatmap: np.ndarray, sensors: List = None, ellipse = None):
        self.heatmap: np.ndarray = heatmap  # Representation of sheet with height values (sensor readings)

        self.sensors: List = sensors        # Sensors coordinates (for scatter plot)

        self.ellipse = ellipse

        X, Y = self.create_xy(heatmap)
        self.X: np.ndarray = X              # Xs of mesh
        self.Y: np.ndarray = Y              # Ys of mesh

    def plot(self) -> None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # map the data to rgba values from a colormap
        colors = cm.ScalarMappable(cmap=cm.coolwarm).to_rgba(self.heatmap)

        # plot_surface with points X,Y,Z and data_value as colors
        surf = ax.plot_surface(self.X, self.Y, self.heatmap, rstride=1, cstride=1, facecolors=colors,
                               linewidth=0, antialiased=True)

        if self.sensors:
            scatter = ax.scatter([s.x for s in self.sensors], [s.y for s in self.sensors], [0 for _ in self.sensors],
                                 c=["b" if not self.sensors[i].reading > 0 else "r" for i in
                                    range(len(self.sensors))], s=[50 for _ in self.sensors])

        if self.ellipse:
            patch = Ellipse(xy=(self.ellipse.h, self.ellipse.k), width=self.ellipse.a, height=self.ellipse.b, color="r")
            scatter = ax.add_patch(patch)
            art3d.pathpatch_2d_to_3d(patch)

        plt.show()

    @staticmethod
    def create_xy(heatmap: np.ndarray) -> (np.ndarray, np.ndarray):
        x, y = heatmap.shape
        return np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))


# Testing purposes only, called in Simulator to show readings
if __name__ == "__main__":
    hm = np.array(np.random.normal(size=(15, 15)))
    viz = Visualizer(hm, [])
    viz.plot()
