import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools


class DataAnalyst:
    def __init__(self, frame_path):
        self.df = pandas.read_csv(frame_path)
        self.df.dropna(axis=0, how='any', inplace=True)

        self.df.astype({
            'id': int,
            'frame': int,
            'big/small': int,
            'dynamic/static': int,
            'press/tap': int,
            'dangeours/safe': int
            })

        self.df.sort_values(by=['id', 'frame'], inplace=True, ignore_index=True)
        self.layers = ['force_at_surface_matrix', 'displacements_surface', 'displacements', 'displacements_under']
        self.experiments = list(self.df['id'].unique())

        for disp in self.layers:
            self.df[disp] = self.df[disp].apply(convert_to_array)

    def plot_heat(self, layer, experiment):
        idns = self.df['id'].unique()
        if experiment not in idns:
            return None
        e_df = self.df[self.df['id'] == experiment].copy()

        # e_df['ravel'] = e_df[self.layers[layer]].apply(np.ravel)
        # y = np.stack(e_df['ravel'])
        # x = np.arange(y.shape[0])

        x = np.array([np.arange(8) for _ in range(8)]).reshape((8, 8))

        frames = 1 if (e_df['dynamic/static'].iloc[0] == 0) else len(e_df)
        for i in range(frames):
            y = np.abs(e_df[self.layers[layer]].iloc[i])
            if layer == 0:
                vmax = 500
            else:
                vmax = 3000
            sns.heatmap(y, vmin=0, vmax=vmax)
            addition = ' input' if layer == 0 else ' displacement'
            plt.title(f'D: {layer}, E: {experiment}' + addition)
            if not os.path.exists(f'./out/v15/images/{experiment}'):
                os.mkdir(f'./out/v15/images/{experiment}')
            plt.savefig(f'./out/v15/images/{experiment}/{layer}_{i}.png')
            plt.close()

    def plot_sensor(self, experiment, layer, sensor):
        edf = self.df[self.df['id'] == experiment].copy()
        layer_list = list(edf[self.layers[layer]])
        x = []

        for time_step in layer_list:
            sensor_value = time_step[sensor[0], sensor[1]]
            print(sensor_value)
            x.append(sensor_value)

        plt.plot(x)
        plt.title(f'S{sensor}, E{experiment}')
        plt.show()


def convert_to_array(text):
    matrix = np.matrix(text)
    shape = matrix.shape
    div = int(np.sqrt(shape[1]))
    if (shape == (1, 100)) | (shape == (1, 64)):
        return matrix.reshape((div, div))
    else:
        print(text)
        print("-"*20)


if __name__ == '__main__':
    da = DataAnalyst('./out/v15/v_15_smoothout.csv')

    da.df.to_csv('./out/v15/v15_clean.csv', index=True)

    # for i, j in itertools.product(range(8), range(8)):
    #     da.plot_sensor(15, 1, (i, j))

    # for layer, experiment in itertools.product(range(4), da.experiments):
    #     da.plot_heat(layer, experiment)

