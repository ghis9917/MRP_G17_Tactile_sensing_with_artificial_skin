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

import numpy as np
import pandas as pd
from tqdm import tqdm

def clip_matrix(mat, threshold):
    """
    :param mat: Unclipped matrix
    :param threshold: Threshold after which the sensors do not increase their reading

    :return: Clipped matrix
    """
    mat[mat > threshold] = threshold
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
    return mat * relative_displ


def handle_sequence(input, displ1, displ2, displ3, s=2, m=0, o_s=.4, ratio=0.3, threshold=35):
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
        temp_naive = naive_approximation(input[i], ratio)

        temp_fem = clip_matrix(temp_fem, threshold)
        temp_naive = clip_matrix(temp_naive, threshold)

        temp_fem += gaussian_noise(s, temp_fem.shape, m)
        temp_naive += gaussian_noise(s, temp_naive.shape, m)

        temp_fem += offset_mat
        temp_naive += offset_mat

        processed_seq_fem.append(temp_fem)
        processed_seq_naive.append(temp_naive)

    return temporal_padding(processed_seq_fem), temporal_padding(processed_seq_naive)


def str_to_np(s):
    n = np.matrix(s)

    if n.shape == (1, 100):
        n = n.reshape(10, 10)
        n = n[1:-1, 1:-1]
    elif n.shape == (1, 64):
        n = n.reshape((8, 8))
    else:
        n = np.zeros((8,8))
    return n


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
    file = open("./Simulation/out/v15/results_fem.csv", 'a+', newline='')
    write = csv.writer(file)
    write.writerow("id,frame,big/small,dynamic/static,press/tap,dangeours/safe,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37,S38,S39,S40,S41,S42,S43,S44,S45,S46,S47,S48,S49,S50,S51,S52,S53,S54,S55,S56,S57,S58,S59,S60,S61,S62,S63,shape,pressure,velocity".split(","))
    for r in results:
        idn = [r[0]]
        labels = r[1]
        processed_seq_fem = r[2]#[0]
        # processed_seq_naive = r[3]
        shape = r[3]
        pressure = r[4]
        velocity = r[5]
        length = len(processed_seq_fem)
        # print("Length = ", length)
        for i in range(len(processed_seq_fem)):
            sensor_values = processed_seq_fem[i].flatten().tolist()
            row = idn + [i] + labels + sensor_values[0] + [shape, pressure, velocity]
            print(row)
            write.writerow(row)





if __name__ == "__main__":
    s = []
    results_by_idn = []
    df_total = pd.read_csv("Simulation/out/v15/v15_clean.csv")

    df_list = split_by_idn(df_total)
    for df in tqdm(df_list):
        displ_surf_seq = df["displacements_surface"].apply(str_to_np).tolist()
        displ_seq = df["displacements"].apply(str_to_np).tolist()
        displ_und_seq = df["displacements_under"].apply(str_to_np).tolist()
        forc_seq = df["force_at_surface_matrix"].apply(str_to_np).tolist()
        # print(forc_seq)
        idn = df["id"].iloc[0]
        label = list(df[["big/small", "dynamic/static", "press/tap", "dangeours/safe"]].iloc[0])
        shape = df["shape"].iloc[0]
        pressure = df["pressure"].iloc[0]
        velocity = df["velocity"].iloc[0]

        results_by_idn.append([idn, label, forc_seq, shape, pressure, velocity])
        # results_by_idn.append([idn, label, handle_sequence(forc_seq, displ_surf_seq,
        #                                                     displ_seq, displ_und_seq),
        #                                                     shape, pressure, velocity])
        # print(results_by_idn)
    to_cost_to_csv(results_by_idn)



