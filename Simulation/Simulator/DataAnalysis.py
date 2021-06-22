#%%
import time

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools


#%%
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

        self.sensors = [f'S{s}' for s in range(64)]
        self.experiments = list(sorted(self.df['id'].unique()))

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

    def plot_sensor(self, experiment):
        ex_df = self.df[self.df['id'] == experiment]
        frames = list(sorted(ex_df['frame'].unique()))

        X = []
        for frame in frames:
            time_frame = ex_df[ex_df['frame'] == frame]
            sensor_readings = list(time_frame[self.sensors].iloc[0])
            # print(sensor_readings)
            X.append(sensor_readings)
        X = np.array(X)
        print(X.shape)
        plt.plot(X)
        plt.xlabel('Timestep')
        plt.ylabel('Force (N)')
        plt.show()




#%%
def convert_to_array(text):
    matrix = np.matrix(text)
    shape = matrix.shape
    div = int(np.sqrt(shape[1]))
    if (shape == (1, 100)) | (shape == (1, 64)):
        return matrix.reshape((div, div))
    else:
        print(text)
        print("-"*20)

#%%
if __name__ == '__main__':
    da = DataAnalyst('/Users/abeldewit/Documents/GitHub/MRP_G17_Tactile_sensing_with_artificial_skin/Simulation/out/v15/results_fem_3.csv')

    for exp in da.experiments:
        da.plot_sensor(exp)
        time.sleep(1)

