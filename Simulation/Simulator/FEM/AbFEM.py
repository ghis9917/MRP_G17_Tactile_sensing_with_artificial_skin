import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
import cv2
import os
import Simulation.Utils.Constants as Const
import matplotlib.pyplot as plt
from Simulation.Simulator.FEM.SkinModel import SkinModel, DEFAULT_PLATE, DEFAULT_BEAM


def run_fem(image, max_force=10, mesh_size=5.0, layers=2, vis=True, plate_dict=None, connect_dict=None):
    """
        Runs a single instance of the FEM with an image as input

    :param image: The 'force' image
    :param max_force: The maximum force that is assigned to the highest image value (255)
    :param mesh_size: The coarseness of the plates
    :param layers: The amount of layers on top of each other
    :param vis: Whether to visualize the end displacement
    :param plate_dict: A dictionary with the layers that have plates and

    :return: The results of the FEM analysis
    """

    tmp_skin = SkinModel()                                      # Instantiate the model
    size = (image.shape[0], image.shape[1])                     # Get the size of the model based on the image
    tmp_skin.create_nodes(size[0], size[1], layers, mesh_size)  # Create the nodes for our model

    # For each layer, create plates with their corresponding properties
    if plate_dict is None:
        plate_dict = {k: DEFAULT_PLATE for k in range(layers-1)}
    for key, value in plate_dict.items():
        tmp_skin.create_plates(plate_layer=key, properties=value)

    for layer in range(1, layers):
        tmp_skin.connect_layers(layer - 1, layer, connection_type='Beam')

    tmp_skin.define_support()

    tmp_skin.input_to_load(image, max_force)
    tmp_skin.analyse()

    displacement = tmp_skin.get_all_displacements()

    if vis:
        tmp_skin.visualise()

    return displacement


def sequential_fem(image_array, max_force=100, normalize=False, size=(100, 100), mesh_size=5.0, layers=2, vis=True):
    """
    Takes an array of images, creates a FEM and analyses it in sequence

    :param image_array: List of images
    :param max_force: Maximum force assigned to max image value (255)
    :param normalize: Reshape images to given size
    :param size: Size to reshape to
    :param mesh_size: The coarseness of the skin plates
    :param layers: The amount of layers on top of each-other
    :param vis: Visualize the last iteration or not
    :return: List of analysis results
    """
    total_time = len(image_array)

    if normalize:
        # TODO normalize images to given size
        print(size)
        pass

    for count, image in enumerate(image_array):
        if vis:
            if count == total_time-1:
                run_fem(image, max_force=max_force, mesh_size=mesh_size, layers=layers, vis=True)

        run_fem(image, max_force=max_force, mesh_size=mesh_size, layers=layers, vis=False)

    return None  # Return results of sequential FEM


def read_sequence(path, file_pattern):
    """
    Reads in a sequence of images from a certain folder

    :param path: The path to the folder
    :param file_pattern: The pattern of the image's name
    :return: An array of images
    """
    files_in_path = os.listdir(path)

    files_names = []
    if file_pattern in str(files_in_path):
        for file in files_in_path:
            if file_pattern in file:
                files_names.append(file)

    files = []
    for file in files_names:
        f = cv2.imread(path+file, cv2.IMREAD_GRAYSCALE)
        files.append(f)

    return files


if __name__ == '__main__':
    # Ask user for input
    image_pattern = input("Sequence name: ")
    seq_or_not = input("Sequence (y/n): ")
    ms = float(input("Mesh size: "))

    # Read in sequence of images
    images = read_sequence('../input/', image_pattern)

    if seq_or_not == 'y':
        # Process images through FEM
        sequential_fem(images, layers=1, mesh_size=ms, vis=True)
    else:
        # Process single image
        run_fem(images[-1], layers=5, max_force=100, mesh_size=ms, vis=True)

    sys.exit()
