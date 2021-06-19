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


import numpy as np


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
    offset_mat = offset_matrix(s, input[0].shape, m)
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

    return processed_seq_fem, processed_seq_naive
