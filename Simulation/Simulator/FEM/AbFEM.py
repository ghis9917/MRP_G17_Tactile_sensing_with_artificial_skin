import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
import cv2
import os
import Simulation.Utils.Constants as Const
import matplotlib.pyplot as plt
import pandas
from Simulation.Simulator.FEM.SkinModel import SkinModel, DEFAULT_PLATE, DEFAULT_BEAM


def run_fem(image, max_force=10, mesh_size=5.0, layers=2, vis=True, dict=None,
            cm_size=(20, 20)):
    """
    Runs a single instance of the FEM with an image as input

    :param image: The 'force' image
    :param max_force: The maximum force that is assigned to the highest image value (255)
    :param mesh_size: The coarseness of the plates
    :param layers: The amount of layers on top of each other
    :param vis: Whether to visualize the end displacement
    :param plate_dict: A dictionary with the layers that have plates and their material properties
    :param connect_dict: A dictionary with the beam connections and their material properties
    :param cm_size: Size of the sheet of skin in centimeter

    :return: The results of the FEM analysis
    """
    cm_2_in = 0.3937
    inch_size = [cm_2_in*cm_size[0], cm_2_in*cm_size[1]]

    tmp_skin = SkinModel()                                      # Instantiate the model
    size = (inch_size[0], inch_size[1])             # Get the size of the model based on the image
    tmp_skin.create_nodes(size[0], size[1], layers, mesh_size)  # Create the nodes for our model
    plate_dict = {}
    beam_dict = {}
    ija = IJA(mesh_size)
    for key in dict.keys():
        plate_dict[key] = [dict[key]["t"], dict[key]["E"], dict[key]["nu"]]
        beam_dict[key] = [dict[key]["E"], dict[key]["G"], ija[0], ija[1], ija[2], ija[3]]
    print("PLATE DICT:")
    print(plate_dict)
    print("BEAM DICT:")
    print(beam_dict)
    # For each layer, create plates with their corresponding properties
    print("Creating plates")
    if plate_dict is None:
        plate_dict = {k: DEFAULT_PLATE for k in range(layers-1)}
    for key, value in plate_dict.items():
        # Thickness, E, nu (properties)
        tmp_skin.create_plates(plate_layer=key, properties=value)

    print("Connecting layers")
    for layer in range(1, layers):
        # E, G, Iy, Iz, J, A
        print("LAYERS ", layers)
        tmp_skin.connect_layers(layer - 1, layer, connection_type='Beam', beam_properties=beam_dict[layer-1])

    support_dict = {k: ('None',) for k in range(layers-1)}
    support_dict[layers-1] = ('Fixed',)

    print("Defining support")
    tmp_skin.define_support(support_dict=support_dict)

    print("Adding load")
    tmp_skin.input_to_load(image, max_force, load_type='Plate')

    print("Analysing")
    tmp_skin.analyse()

    print("Getting displacements")
    displacement = tmp_skin.get_all_displacements()
    node_loads = []
    node_titles = []
    for node in tmp_skin.fem.Nodes:
        node_titles.append(node)
        node_loads.append(tmp_skin.fem.GetNode(node).NodeLoads)
    if vis:
        tmp_skin.visualise()
    print(node_titles)
    print(node_loads)
    return displacement, node_titles, node_loads


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


def read_csv(filepath):
    """
    Reads in the material properties for the artificial skin, muscle and fat.
    E = Young's modulus
    nu = Poisson's ratio
    t = thickness in mm

    :return: A dictionary where each key stores a series of properties
        In the shape: {"MaterialName": [E, nu, thickness]}
    """
    dictionary = {}
    csv = pandas.read_csv(filepath)
    for i, row in csv.iterrows():
        dictionary[row["Material"]] = row[1:]

    return dictionary


def keys_to_layers(dicti):
    """
    Method to structure material properties according to the layers that correspond with them.
    Required structure for our FEM simulation.
    :param dicti: A dictionary whose keys should be replaced with their indices
    :return: A dictionary with indices as keys
    """
    layer_props = {}
    index = 0
    for key in dicti.keys():
        layer_props[index] = dicti[key]
        index += 1
    return layer_props


def IJA(x):
    """
    This method calculates the moments of inertia and the cross_sectional surface area required for the FEM
    :param: x : mesh_size in inches
    :return: [Iy, Iz, J, A]
    """

    # Iy = Iz is the x^4 /12. They are equal because the surface is a square, not a rectangle.

    Iy = Iz = np.power(4, x)/12
    J = Iy + Iz
    A = np.square(x)

    return [Iy, Iz, J, A]


if __name__ == '__main__':
    mat_props = read_csv("./FEM/Material Properties (E, nu, G).csv")
    print(mat_props)
    mat_props = keys_to_layers(mat_props)
    print(mat_props.keys())
    print(mat_props[0]["nu"])

    # Ask user for input
    image_pattern = input("Sequence name: ")
    seq_or_not = input("Sequence (y/n): ")
    ms = float(input("Mesh size: "))

    # Read in sequence of images
    images = read_sequence('../../input/', image_pattern)

    if seq_or_not == 'y':
        # Process images through FEM
        sequential_fem(images, layers=1, mesh_size=ms, vis=True, dict=mat_props)

    else:
        # Process single image
        print(images[-1].shape)
        run_fem(images[-1], layers=len(mat_props), max_force=100, mesh_size=ms, vis=True, dict=mat_props)

    sys.exit()
