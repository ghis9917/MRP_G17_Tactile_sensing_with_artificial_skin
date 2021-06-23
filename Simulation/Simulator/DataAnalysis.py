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

from Simulation.Simulator.PostProcessing import str_to_np
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

    def plot_sensor(self, experiment, force_mats=[], pressure=0.0, margin=1):
        name, labels, y_one = self.get_experiment(experiment)

        if margin > 0:
            y_two = y_one.reshape((30, 8, 8))
            y_three = y_two[:, margin:-margin, margin:-margin]
            new_shape = 8 - 2*margin
            y = y_three.reshape((30, new_shape * new_shape))
        else:
            y = y_one

        y = np.amax(y, axis=1)
        force = y / y.max() * pressure

        xvals = np.arange(0, 30)
        f = interp1d(xvals, y, kind='cubic', axis=0)
        xnew = np.arange(0, 29, 0.1)
        ynew = f(xnew)

        # plt.plot(y, 'o', markersize=3)

        plt.plot(force)
        plt.plot(xnew, ynew)
        plt.title(name + ' ' + labels)

        plt.xlabel('Timestep')
        plt.ylabel('Force (N)')
        plt.show()

    def plot_vs_n(self, experiment, pressure=0, margin=1):
        name, labels, y_one = self.get_experiment(experiment)

        if margin > 0:
            y_two = y_one.reshape((30, 8, 8))
            y_three = y_two[:, margin:-margin, margin:-margin]
            new_shape = 8 - 2*margin
            y = y_three.reshape((30, new_shape * new_shape))
        else:
            y = y_one

        new_y = []
        for i in range(y.shape[1]):
            column = y[:, i]
            if column.max() > 10:
                new_y.append(column)
        y = np.array(new_y).T

        if len(y) == 0:
            return

        xvals = np.arange(0, 30)
        # f = interp1d(xvals, y, kind='cubic', axis=0)
        # xnew = np.arange(0, 29, 0.1)
        # ynew = f(xnew)

        # plt.plot(y, 'o', markersize=3)

        plt.plot(xvals, y)
        plt.title(name + ' ' + labels)
        plt.ylim([-1, 16])
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
    da = DataAnalyst('/Users/abeldewit/Documents/GitHub/MRP_G17_Tactile_sensing_with_artificial_skin/Simulation/out/v18/results_fem_19.csv')

    if 'S0' not in da.df.columns:
        print("Only sensor matrices")

        big_sensor_mat = []
        for i in range(len(da.df)):
            sr = convert_to_array(da.df['displacements'].iloc[i])
            big_sensor_mat.append(sr)

        columns = [f'S{i}' for i in range(64)]
        extra_df = pd.DataFrame(big_sensor_mat, columns=columns)

        da.df = pd.concat([da.df, extra_df], axis=1).reset_index()

    raw_data = pd.read_csv('/Users/abeldewit/Documents/GitHub/MRP_G17_Tactile_sensing_with_artificial_skin/Simulation/out/v18/v_18_smoothout.csv')
    shape = 'circle blur.png'  # input("Shape: ")

    sampling = False
    if shape == '':
        sampling = True

    for _ in range(100):
        if sampling:
            samp_df = da.df
        else:
            samp_df = da.df[da.df['shape'] == shape]

        ran_num = random.randint(0, len(samp_df))
        ran_idx = samp_df.index[ran_num]
        random_row = samp_df.loc[ran_idx]

        sample = int(random_row['id'])
        pressure = float(random_row['pressure'])
        raw_samp = raw_data[raw_data['id'] == sample]

        force_mats = []
        for i in range(len(raw_samp)):
            force_data = str(raw_samp['force_at_surface_matrix'].iloc[i])
            matrix = str_to_np(force_data)
            force_mats.append(matrix)

        # da.plot_sensor(sample, margin=0, pressure=pressure)
        da.plot_vs_n(sample, margin=0, pressure=pressure)
        # da.plot_3d(exp)
        # da.plot_heat(exp)

        time.sleep(1)

