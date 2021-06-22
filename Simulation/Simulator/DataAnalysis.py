#%%
import random
import re
import time

import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import itertools
from scipy.interpolate import interp1d
import cv2

from Simulation.Visualization.Visualizer import Visualizer


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

    def plot_heat(self, experiment):
        name, labels, y = self.get_experiment(experiment)

        x = np.array([np.arange(8) for _ in range(8)]).reshape((8, 8))

        heat_y = np.array(y).reshape((len(y), 8, 8))
        vmax = np.array(y).max()

        frames = len(y)
        for i in range(frames):
            sns.heatmap(heat_y[i, :, :], vmin=0.0, vmax=vmax)
            plt.title(f'E: {experiment} {name} {labels}')
            # if not os.path.exists(f'./out/v15/images/{experiment}'):
            #     os.mkdir(f'./out/v15/images/{experiment}')
            # plt.savefig(f'./out/v15/images/{experiment}/{layer}_{i}.png')
            plt.show()
            plt.close()

    def plot_sensor(self, experiment):
        name, labels, y = self.get_experiment(experiment)

        xvals = np.arange(0, 30)
        f = interp1d(xvals, y, kind='quadratic', axis=0)
        xnew = np.arange(0, 29, 0.1)
        ynew = f(xnew)

        # plt.plot(y, 'o', markersize=3)

        plt.plot(xnew, ynew)
        plt.title(name + ' ' + labels)
        plt.ylim([-2, 16])
        plt.xlabel('Timestep')
        plt.ylabel('Force (N)')
        plt.show()

    def plot_3d(self, experiment):
        name, labels, y = self.get_experiment(experiment)

        sensor_map = np.ones(shape=(8, 8))

        stupid_list = []
        for frame in y:
            x = frame.reshape((8, 8))
            stupid_list.append(x)

        counter = 0

        resized_outputs = [cv2.resize(output, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for output in
                           stupid_list]
        viz1 = Visualizer(size=(250, 250))
        viz1.ani_3D(resized_outputs,
                    cv2.resize(sensor_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
        viz1.ani_2D(resized_outputs,
                    cv2.resize(sensor_map, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
        counter += 1

    def get_experiment(self, experiment):
        ex_df = self.df[self.df['id'] == experiment]
        frames = list(sorted(ex_df['frame'].unique()))

        name = str(ex_df['shape'].iloc[0]).split('.')[0].replace('_', ' ')
        labels = str(list(ex_df[['big/small', 'dynamic/static', 'press/tap', 'dangeours/safe']].astype(int).iloc[0]))

        y = []
        for frame in frames:
            time_frame = ex_df[ex_df['frame'] == frame]
            sensor_readings = np.array(list(time_frame[self.sensors].iloc[0]))
            y.append(sensor_readings)
        y = np.array(y).astype(float)
        return name, labels, y

#%%
def convert_to_array(text):
    pattern = re.compile('(\d{1,2}.\d+e(\+|-)\d\d)')
    match = re.findall(r'(\d{1,2}.\d+e(\+|-)\d\d)', text)

    sr = []
    for m in match:
        sensor = m[0]
        sr.append(sensor)
    return sr


#%%
if __name__ == '__main__':
    da = DataAnalyst('/Users/abeldewit/Documents/GitHub/MRP_G17_Tactile_sensing_with_artificial_skin/Simulation/out/v18/results_fem_18.csv')

    if 'S0' not in da.df.columns:
        print("Only sensor matrices")

        big_sensor_mat = []
        for i in range(len(da.df)):
            sr = convert_to_array(da.df['displacements'].iloc[i])
            big_sensor_mat.append(sr)

        columns = [f'S{i}' for i in range(64)]
        extra_df = pd.DataFrame(big_sensor_mat, columns=columns)

        da.df = pd.concat([da.df, extra_df], axis=1).reset_index()

    shape = input("Shape: ")

    sampling = False
    if shape == '':
        sampling = True

    for _ in range(1):
        if sampling:
            sample = random.randint(0, len(da.experiments))
        else:
            samp_df = da.df[da.df['shape'] == shape]
            # samp_df = samp_df[samp_df['dynamic/static'] == 1]
            rannum = random.randint(samp_df.index.min(), samp_df.index.max())
            sample = int(samp_df.iloc[random.randint(0, len(samp_df))]['id'])

        exp = da.experiments[sample]
        # da.plot_sensor(exp)
        da.plot_3d(exp)
        # da.plot_heat(exp)

        time.sleep(10)

