from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:

    def __init__(self, heatmap: np.ndarray):
        self.heatmap = heatmap
        self.create_xy(heatmap)

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # map the data to rgba values from a colormap
        colors = cm.ScalarMappable(cmap=cm.coolwarm).to_rgba(self.heatmap)

        # plot_surface with points X,Y,Z and data_value as colors
        surf = ax.plot_surface(self.X, self.Y, self.heatmap, rstride=1, cstride=1, facecolors=colors,
                               linewidth=0, antialiased=True)

        plt.show()

    def create_xy(self, heatmap: np.ndarray):
        x, y = heatmap.shape
        self.X, self.Y = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))


if __name__ == "__main__":

    hm = np.array(np.random.normal(size=(15, 15)))

    viz = Visualizer(hm)
    viz.plot()
