from typing import List

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib import cm, animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import Simulation.Utils.Constants as Const


class Visualizer:

    def __init__(self, size: tuple = (8, 8)):

        X, Y = self.create_xy(size)
        self.X: np.ndarray = X  # Xs of mesh
        self.Y: np.ndarray = Y  # Ys of mesh

    def ani_2D(self, animation_heatmap, sensors_map):
        fig = plt.figure()
        ax = fig.add_subplot()

        plot_sensors_reading = ax.imshow(animation_heatmap[0].T+animation_heatmap[0].T*sensors_map.T, cmap='gray')

        vmin = np.array(animation_heatmap).min()
        vmax = np.array(animation_heatmap).max()

        def data_gen(framenumber, soln, plot2, X, Y):
            ax.clear()
            plot2 = ax.imshow(soln[framenumber].T+animation_heatmap[framenumber].T*sensors_map.T,
                              cmap='gray',
                              vmin=vmin,
                              vmax=vmax
            )
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

        anim.save(f'./out/2D.gif', fps=5, dpi=80)

    def ani_3D(self, animation_heatmap, sensors_map):
        import pathlib
        print(pathlib.Path().absolute())

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plot_args = {
            'rstride': 1,
            'cstride': 1,
            'cmap': cm.bwr,
            'linewidth': 0.01,
            'antialiased': True,
            'color': 'w',
            'shade': True,
            'vmin': np.array(animation_heatmap).min(),
            'vmax': np.array(animation_heatmap).max()
        }

        plot_grid = ax.plot_surface(self.X, self.Y, animation_heatmap[0].T, **plot_args)
        plot_sensors_reading = ax.plot_surface(self.X, self.Y, animation_heatmap[0].T*sensors_map.T, **plot_args)
        ax.set_zlim3d(0, Const.MAX_FORCE)

        def data_gen(framenumber, soln, plot1, plot2, X, Y):
            ax.clear()
            print('3D', framenumber)
            plot1 = ax.plot_surface(X, Y, soln[framenumber].T, **plot_args)
            plot2 = ax.plot_surface(X, Y, soln[framenumber].T*sensors_map.T, **plot_args)
            ax.set_zlim3d(0, np.max(np.array(soln)))
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

        anim.save('./out/3D.gif', fps = 5, dpi = 80)

    @staticmethod
    def create_xy(heatmap: tuple) -> (np.ndarray, np.ndarray):
        x, y = heatmap
        return np.meshgrid(np.arange(x), np.arange(y))


# Testing purposes only, called in Simulator to show readings
if __name__ == "__main__":
    hm = np.array(np.random.normal(size=(15, 15)))
    viz = Visualizer(hm, [])
    viz.plot()
