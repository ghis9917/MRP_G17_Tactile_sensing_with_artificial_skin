from typing import List

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib import cm, animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import Simulation.Utils.Constants as Const


class Visualizer:

    def __init__(self, heatmap: np.ndarray, sensors: np.ndarray = None, ellipse=None):
        self.heatmap: np.ndarray = heatmap.T  # Representation of sheet with height values (sensor readings)

        self.sensors: List = sensors.T  # Sensors coordinates (for scatter plot)

        self.ellipse = ellipse

        X, Y = self.create_xy(heatmap)
        self.X: np.ndarray = X  # Xs of mesh
        self.Y: np.ndarray = Y  # Ys of mesh

    def create_image(self) -> None:
        self.heatmap = np.interp(self.heatmap, (self.heatmap.min(), self.heatmap.max()), (0, +255)).round()
        self.sensors = np.interp(self.sensors, (self.sensors.min(), self.sensors.max()), (0, +255)).round()
        heatmap_img = Image.fromarray(np.uint8(self.heatmap), 'L')
        sensors_img = Image.fromarray(np.uint8(self.sensors), 'L')
        heatmap_img.save('out/HeatMap.jpg')
        sensors_img.save('out/SensorsOnly.jpg')

    def plot(self) -> None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # map the data to rgba values from a colormap
        colors = cm.ScalarMappable(cmap=cm.coolwarm).to_rgba(self.heatmap)

        # plot_surface with points X,Y,Z and data_value as colors
        ax.plot_surface(self.X, self.Y, self.heatmap, rstride=1, cstride=1, facecolors=colors,
                        linewidth=0, antialiased=True)

        if self.sensors is not None:
            ax.plot_surface(self.X, self.Y, self.sensors, rstride=1, cstride=1, facecolors=colors,
                            linewidth=0, antialiased=True)

        if self.ellipse is not None:
            patch = Ellipse(xy=(self.ellipse.h, self.ellipse.k), width=self.ellipse.a, height=self.ellipse.b, color="r")
            ax.add_patch(patch)
            art3d.pathpatch_2d_to_3d(patch)

        plt.show()

    def ani_2D(self, animation_heatmap, sensors_map):
        fig = plt.figure()
        ax = fig.add_subplot()

        plot_sensors_reading = ax.imshow(animation_heatmap[0].T+animation_heatmap[0].T*sensors_map.T, cmap='gray')

        def data_gen(framenumber, soln, plot2, X, Y):
            ax.clear()
            plot2 = ax.imshow(soln[framenumber].T+animation_heatmap[framenumber].T*sensors_map.T, cmap='gray')
            return plot2,

        anim = animation.FuncAnimation(
            fig,
            data_gen,
            frames=len(animation_heatmap),
            fargs=(animation_heatmap, plot_sensors_reading, self.X, self.Y),
            interval=10,
            blit=False,
            repeat=True
        )

        anim.save('../out/2D.gif', fps = 60, dpi = 80)
        plt.show()

    def ani_3D(self, animation_heatmap, sensors_map):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_args = {
            'rstride': 1,
            'cstride': 1,
            'cmap': cm.bwr,
            'linewidth': 0.01,
            'antialiased': True,
            'color': 'w',
            'shade': True
        }

        plot_grid = ax.plot_surface(self.X, self.Y, animation_heatmap[0].T, **plot_args)
        plot_sensors_reading = ax.plot_surface(self.X, self.Y, animation_heatmap[0].T*sensors_map.T, **plot_args)
        ax.set_zlim3d(0, Const.MAX_FORCE)

        def data_gen(framenumber, soln, plot1, plot2, X, Y):
            ax.clear()
            print(framenumber)
            plot1 = ax.plot_surface(X, Y, soln[framenumber].T, **plot_args)
            plot2 = ax.plot_surface(X, Y, soln[framenumber].T*sensors_map.T, **plot_args)
            ax.set_zlim3d(0, Const.MAX_FORCE)
            return (plot1, plot2),

        anim = animation.FuncAnimation(
            fig,
            data_gen,
            frames=len(animation_heatmap) - 1,
            fargs=(animation_heatmap, plot_grid, plot_sensors_reading, self.X, self.Y),
            interval=10,
            blit=False,
            repeat=True
        )

        anim.save('../out/3D.gif', fps = 60, dpi = 80)
        plt.show()

    @staticmethod
    def create_xy(heatmap: np.ndarray) -> (np.ndarray, np.ndarray):
        x, y = heatmap.shape
        return np.meshgrid(np.arange(x), np.arange(y))


# Testing purposes only, called in Simulator to show readings
if __name__ == "__main__":
    hm = np.array(np.random.normal(size=(15, 15)))
    viz = Visualizer(hm, [])
    viz.plot()
