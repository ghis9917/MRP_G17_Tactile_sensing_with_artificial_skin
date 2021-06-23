"""
Class that handles the rough simulation data in order to simulate sensor readings
Workings:
    Clipping: The sensors have a maximum value of x N, they will not read any value above this threshold, with the exception of some noise.
    Noise: The sensor readings are not perfect and are subject to some noise. For robustness, we simulate this noise according to some distribution.
            For simplicity's sake we assume a gaussian
    Approximation based on FEM -> use displacements to approximate the force that is read by the sensors (which are inside the skin)
    Naive approximation based on experiments on the prototype -> evaluate the experiment data and figure out a ratio or nunmber that
            can be used to create a similar force reduction at the sensor level.
"""
import csv
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

def clip_matrix(mat, threshold):
    """
    :param mat: Unclipped matrix
    :param threshold: Threshold after which the sensors do not increase their reading

    :return: Clipped matrix
    """
    # mat[mat > threshold] = threshold
    mat = np.clip(mat, a_min=-1, a_max=15)
    return mat


def gaussian_noise(s, shape, m=0):
    """
    Returns gaussian noise matrix
    :param m: Mean of the noise, default: m=0
    :param s: Standard deviation of the noise function
    :param shape: The shape you want the noise matrix to be

    :return: A matrix with noise values
    """
    return np.random.normal(m, s, shape)


def offset_matrix(s, shape, m=0):
    """
    Returns an offset matrix. Essentially the same method as gaussian_noise().
    :param m: Mean of the noise distribution, default: m=0
    :param s: Standard deviation of the noise function
    :param shape: The shape you want the offset matrix to be

    :return: A matrix with offset values
    """
    return np.random.normal(m, s, shape)


def naive_approximation(mat, ratio):
    """
    An approximation based on the experiment results of the prototype.
    (Assumes identical sensor depth to the prototype)
    :param mat: Input matrix
    :param ratio: The ratio input matrix can be multiplied with that is found to result in a good approximation of the
        resulting force at the depth of the sensors.

    :return: The approximated sensor reading matrix
    """
    return mat*ratio


def fem_approximation(mat, displ0, target_displ, displ2):
    """
    Approximates remaining force at sensor level with use of fem. Concept: displacement is a metric for absorption,
    assumption: this absorption also corresponds to the force that is absorbed. Press against metal -> metal presses
    back with almost the same force. Press against silicone -> shock is absorbed.
    :param mat: Input matrix
    :param displ0: Displacement matrix of the surface layer
    :param target_displ: Displacement matrix of the sensor layer
    :param displ2: Displacement matrix of the bottom layer

    :return:
    """

    total_displ = displ0 + target_displ + displ2
    relative_displ = target_displ / total_displ
    return mat * np.nan_to_num(relative_displ)


def handle_sequence(input, displ1, displ2, displ3, s=.1, m=0, o_s=.05, ratio=0.3, threshold=10):
    """
    Handles a sequence of inputs and displacements and applies all post_processing steps.
    Assumes the same matrix shape for each instance of all 4 input sequences.

    :param input: Sequence of images that were used as input for the FEM (normalized etc)
    :param displ1: Sequence of displacement matrices of the surface layer
    :param displ2: Sequence of displacement matrices of the first layer
    :param displ3: Sequence of displacement matrices of the second layer
    :param s: Standard deviation for the gaussian noise
    :param m: Median for the noise distributions (I think this should stay at 0)
    :param o_s: Standard deviation for the offset noise
    :param ratio: The ratio of remaining force found in the experiments
    :param threshold: The threshold at which the sensors stop working

    :return: sequence of processed matrices (fem), sequence of processed matrices (naive)
    """
    processed_seq_fem = []
    processed_seq_naive = []
    offset_mat = offset_matrix(o_s, (8, 8), m)
    for i in range(len(input)):
        temp_fem = fem_approximation(input[i], displ1[i], displ2[i], displ3[i])
        check_nan(temp_fem)
        temp_naive = naive_approximation(input[i], ratio)

        # temp_fem += offset_mat
        # check_nan(temp_fem)
        # temp_naive += offset_mat

        # temp_fem += gaussian_noise(s, temp_fem.shape, m)
        # check_nan(temp_fem)
        # temp_naive += gaussian_noise(s, temp_naive.shape, m)

        temp_fem = clip_matrix(temp_fem, threshold)
        check_nan(temp_fem)
        temp_naive = clip_matrix(temp_naive, threshold)

        processed_seq_fem.append(temp_fem)
        processed_seq_naive.append(temp_naive)

    processed_seq_fem, processed_seq_naive = temporal_padding(processed_seq_fem), temporal_padding(processed_seq_naive)

    processed_seq_fem = [x + gaussian_noise(s, x.shape, m) + offset_mat for x in processed_seq_fem]

    processed_seq_naive = [x + gaussian_noise(s, x.shape, m) + offset_mat for x in processed_seq_naive]
    return processed_seq_fem, processed_seq_naive


def check_nan(matrix):
    if matrix.any() == np.nan:
        print("nan found")


def str_to_np(s):
    pattern = re.compile('(\d{1,2}.\d+e(\+|-)\d\d)')
    match = re.findall(r'(\d{1,2}.\d*(e*(\+|-)\d\d)*)', s)

    sr = []
    for m in match:
        sensor = m[0]
        if sensor == '':
            sensor = m[2]
        sr.append(float(sensor))
    one_d = np.array(sr)
    if len(one_d) == 64:
        one_d = one_d.reshape((8, 8))
    elif len(one_d) == 100:
        one_d = one_d.reshape((10, 10))
        one_d = one_d[1:-1, 1:-1]
    else:
        one_d = np.zeros((8, 8))
    return one_d



def temporal_padding(seq):
    l = len(seq)
    index = np.random.randint(30-l)
    # print("INDEX = ", index)
    for i in range(30-l):
        if i < index:
            el = seq[0]
            seq.insert(0, el*0.1)
        elif i >= index:
            el = seq[i+l-1]
            seq.append(el*0.1)

    # print("LEN = ", len(seq))
    return seq


def split_by_idn(df):
    l = []
    names = df['id'].unique().tolist()  # find unique values
    for n in names:
        l.append(df[df['id'] == n])
    return l


def to_cost_to_csv(results):
    # id,frame,big/small,dynamic/static,press/tap,dangeours/safe, sensors
    # id,frame,big/small,dynamic/static,press/tap,dangeours/safe,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37,S38,S39,S40,S41,S42,S43,S44,S45,S46,S47,S48,S49,S50,S51,S52,S53,S54,S55,S56,S57,S58,S59,S60,S61,S62,S63, shape,pressure,velocity
    file = open("./Simulation/out/v18/results_fem_19.csv", 'w+')
    write = csv.writer(file)
    write.writerow("id,frame,big/small,dynamic/static,press/tap,dangeours/safe,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37,S38,S39,S40,S41,S42,S43,S44,S45,S46,S47,S48,S49,S50,S51,S52,S53,S54,S55,S56,S57,S58,S59,S60,S61,S62,S63,shape,pressure,velocity".split(","))
    for r in results:
        write.writerow(r)


if __name__ == "__main__":
    s = []
    results_by_idn = []
    df_total = pd.read_csv("Simulation/out/v18/v_18_smoothout.csv")
    df_total.dropna(how='any', inplace=True)

    df_list = split_by_idn(df_total)

    for idn, df in tqdm(enumerate(df_list)):
        displ_surf_seq = df["displacements_surface"].apply(str_to_np).tolist()
        displ_seq = df["displacements"].apply(str_to_np).tolist()
        displ_und_seq = df["displacements_under"].apply(str_to_np).tolist()
        forc_seq = df["force_at_surface_matrix"].apply(str_to_np).tolist()
        # print(forc_seq)

        label = list(df[["big/small", "dynamic/static", "press/tap", "dangeours/safe"]].iloc[0])
        shape = df["shape"].iloc[0]
        pressure = df["pressure"].iloc[0]
        velocity = df["velocity"].iloc[0]

        new_id_num = int(df['id'].iloc[0])

        # temp_pad = temporal_padding(forc_seq)

        temp_pad = handle_sequence(forc_seq, displ_surf_seq, displ_seq, displ_und_seq)[0]

        for frame, matrix in enumerate(temp_pad):
            new_row = [new_id_num, frame]
            new_row.extend(label)

            new_row.extend(matrix.flatten().tolist())

            new_row.extend([shape, str(pressure), str(velocity)])
            results_by_idn.append(new_row)
        # results_by_idn.append([idn, label, ,
        #                                                     shape, pressure, velocity])
        # print(results_by_idn)
    to_cost_to_csv(results_by_idn)

    # df = pd.read_csv("./Simulation/out/v18/results_fem_18.csv")
    print("end!")



