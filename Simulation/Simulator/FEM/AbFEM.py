import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


class SkinModel:
    def __init__(self):
        self.fem = FEModel3D()
        self.t = 1
        self.E = 0.001
        self.nu = 0.17

        self.width = None
        self.height = None
        self.layers = None
        self.mesh_size = None
        self.xlsp = None
        self.ylsp = None

        self.node_name_list = []
        self.node_matrix = None
        self.plate_matrix = None
        self.factors = None

    def create_nodes(self, width, height, layers=1, mesh_size=1.0):
        """
        Creates nodes based on the specified width and height of the skin divided by the mesh size

        :param width: Width of the skin sheet
        :param height: Height of the skin sheet
        :param layers: Amount of node layers
        :param mesh_size: Subdivision of the sheet
        :return: None
        """
        self.width, self.height = width, height
        self.layers, self.mesh_size = layers, mesh_size
        
        # Width and height divided by the mesh size to get our node step size
        x_steps = int(width // mesh_size)
        y_steps = int(height // mesh_size)
        
        # Create a list with each x and y value for the possible nodes
        x_lsp = np.linspace(0, width, x_steps+1)
        self.xlsp = x_lsp
        y_lsp = np.linspace(0, height, y_steps+1)
        self.ylsp = y_lsp

        layer_distance = x_steps if x_steps > y_steps else y_steps
        
        # Creates a 3D array of strings to access the nodes based on their x, y, z location
        layer_vec = ["" for _ in range(self.layers)]
        x_mat = [layer_vec for _ in range(len(self.xlsp))]
        self.node_matrix = np.array([x_mat for _ in range(len(self.ylsp))], dtype=object)

        # For each layer, we construct a combination of possible x and y values
        for layer in range(layers):
            for x, y in itertools.product(range(len(x_lsp)), range(len(y_lsp))):
                x_cord = x_lsp[x]
                y_cord = y_lsp[y]

                # Create a node with it's coordinates N{x}.{y}.{z}
                name = f'N{x}.{y}.{layer}'

                # Add the node to the FEM, a node list and the 3D matrix
                self.fem.AddNode(name, x_cord, y_cord, layer*-20)
                self.node_name_list.append(name)
                self.node_matrix[x, y, layer] = name

        # Convert the node matrix after creation to a numpy array for easy slicing
        self.node_matrix = np.array(self.node_matrix, dtype=object)

    def create_plates(self):
        """
        This method creates plates between each combination of four nodes in the existing node matrix,
        in order to create a 'layer' of the material that we provide for the plates.

        :return: None
        """
        if (self.width is None) | (self.height is None) | (self.layers is None):
            raise ValueError("Tried to add plates without nodes")

        shap = self.node_matrix.shape
        self.plate_matrix = np.array([[["" for _ in range(shap[2])] for _ in range(shap[1]-1)] for _ in range(shap[0]-1)], dtype=object)
        len_layer = self.layers - 1 if self.layers > 1 else 1
        for layer in range(len_layer):
            for i, j in itertools.product(range(len(self.xlsp) - 1), range(len(self.ylsp) - 1)):

                n1 = 'N' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(layer))
                n2 = 'N' + str(int(i)) + '.' + str(int(j+1)) + '.' + str(int(layer))
                n3 = 'N' + str(int(i+1)) + '.' + str(int(j)) + '.' + str(int(layer))
                n4 = 'N' + str(int(i+1)) + '.' + str(int(j+1)) + '.' + str(int(layer))

                name = 'P' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(layer))

                self.fem.AddPlate(name, n1, n2, n4, n3, self.t, self.E, self.nu)

                self.plate_matrix[i, j, layer] = name

    def connect_layers(self, l0, l1, type='Beam'):
        """
        Connect the different layers of nodes with either a 'Beam' or a 'Plate'

        :param l0: First layer to connect
        :param l1: Second layer to connect
        :param type: Beam or Plate
        :return: None
        """
        # If the type is Beam, we can create a 1-1 mapping between the nodes
        if type == 'Beam':
            layer_0 = self.node_matrix[:, :, l0].ravel()
            layer_1 = self.node_matrix[:, :, l1].ravel()

            for id in range(len(layer_0)):
                n1 = layer_0[id]
                n2 = layer_1[id]

                self.fem.AddMember(f'M{l0}.{l1}.{id}', n1, n2, 29000, 11400, 100, 150, 250, 20)

        # Connect each combination of (nearby) nodes with plates
        if type == 'Plate':
            layer_0 = self.node_matrix[:, :, l0]
            layer_1 = self.node_matrix[:, :, l1]

            visited = []
            one_layer_tuple = []

            for i, j in itertools.product(range(layer_0.shape[0]), range(layer_0.shape[1])):

                others = []
                if i > 0:
                    others.append((i-1, j))
                if j > 0:
                    others.append((i, j))
                if i < layer_0.shape[0]-1:
                    others.append((i+1, j))
                if j < layer_0.shape[1]-1:
                    others.append((i, j+1))
                    
                for o in others:
                    if o in visited:
                        others.remove(o)

                one_layer_tuple.extend([((i, j), o) for o in others])
                visited.append((i, j))

            for name, combination in enumerate(one_layer_tuple):
                c1 = combination[0]
                c2 = combination[1]

                n0 = f'N{c1[0]}.{c1[1]}.{l0}'
                n1 = f'N{c2[0]}.{c2[1]}.{l0}'

                n2 = f'N{c1[0]}.{c1[1]}.{l1}'
                n3 = f'N{c2[0]}.{c2[1]}.{l1}'

                plate_name = f'P.{name}.Middle'

                self.fem.AddPlate(plate_name, n0, n1, n3, n2, self.t, self.E, self.nu)

                # Not added to plate matrix?

    def get_node_mat(self):
        return self.node_matrix

    def input_to_load(self, img, max_force):
        """
        Divides the input image in such a way the intensity of the image's values can be translated to pressures
        on nodes or plates

        :param img: The input image with white levels indicating the amount of force
        :param max_force: The image's range [0-255] will be mapped to the force range [0-max_force]
        :return: None
        """
        # For now, 100 by 100 input, 10 by 10 nodes -> window = -5 to +5
        # So, for index i the window would be from i*10 to i*10+10

        x_step = self.xlsp[1]
        y_step = self.ylsp[1]

        s = self.plate_matrix.shape
        print(s)

        for x, y in itertools.product(range(s[0]), range(s[1])):
            x_0 = int(x*x_step)
            x_1 = int(x*x_step + y_step)
            y_0 = int(y*y_step)
            y_1 = int(y*y_step + y_step)

            window = img[x_0:x_1, y_0:y_1]
            avg = np.mean(window)

            if avg != 0.0:
                force = (avg / 255) * max_force

                plat_name = self.plate_matrix[x, y, 0]

                case_name = 'Case 1'  # +plat_name[1:]
                self.fem.Plates[plat_name].pressures.append([force, case_name])
                # self.fem.AddNodeLoad(self.node_matrix[x, y, 0], 'FZ', -force)

    def __define_support(self, type="Pinned", loc="Sides"):
        """
        Sets the nodes on the edges of the skin to be a support, disallowing either movement, rotation or both

        :param type: Pinned = no movement, Fixed = no movement, no rotation
        :return: None
        """
        if loc == "Sides":
            for l in range(self.layers):
                nodes = []
                if l < self.layers-1:
                    pass
                    # nodes.extend(self.node_matrix[:, 0, l])
                    # nodes.extend(self.node_matrix[:, -1, l])
                    # nodes.extend(self.node_matrix[0, :, l])
                    # nodes.extend(self.node_matrix[-1, :, l])
                else:
                    nodes = self.node_matrix[:, :, l].ravel()
                    print(nodes)

                for n in nodes:
                    node = self.fem.Nodes[n]
                    if type == "Pinned":
                        node.SupportDX, node.SupportDY, node.SupportDZ = True, True, True
                    elif type == "Fixed":
                        node.SupportDX, node.SupportDY, node.SupportDZ, node.SupportRX, node.SupportRY, node.SupportRZ = True, True, True, True, True, True
        if loc == "All":
            for node in self.fem.Nodes.values():
                node.SupportDX, node.SupportDY, node.SupportDZ = True, True, True

    def get_model(self):
        return self.fem

    def analyse(self):
        self.__define_support(type='Pinned')

        self.fem.Analyze(check_statics=True, sparse=True, max_iter=30)

    def visualise(self):
        Visualization.RenderModel(self.fem,
                                  text_height=0.5,
                                  deformed_shape=False,
                                  deformed_scale=300,
                                  render_loads=True,
                                  color_map='dz',
                                  case=None,
                                  combo_name='Combo 1')
    
    def plot_forces(self, force_type):

        breakpoint()

    def get_node_displacement(self, i, j, disp='DZ'):
        node_name = self.node_matrix[i, j]
        node = self.fem.Nodes(node_name)

        if disp == 'DZ':
            return node.DZ

    def get_all_displacements(self):
        xsize = self.node_matrix.shape[0]
        ysize = self.node_matrix.shape[1]
        displacement_mat = np.zeros(shape=(xsize, ysize))
        for i, j in itertools.product(range(xsize), range(ysize)):
            t_name = self.node_matrix[i, j]
            tmp_dz = self.fem.Nodes[t_name].DZ
            displacement_mat[i, j] = tmp_dz
        return displacement_mat


def run_fem(image, max_force=100, mesh_size=5.0, layers=2, vis=True):
    """
    Runs a single instance of the FEM with an image as input

    :param image: The 'force' image
    :param max_force: The maximum force that is assigned to the highest image value (255)
    :param mesh_size: The coarseness of the plates
    :param layers: The amount of layers on top of each other
    :param vis: Whether to visualize the end displacement
    :return: The results of the FEM analysis
    """
    tmp_skin = SkinModel()  # Instantiate the model

    size = (image.shape[0], image.shape[1])

    tmp_skin.create_nodes(size[0], size[1], layers, mesh_size)
    tmp_skin.create_plates()

    if layers > 1:
        pass
        tmp_skin.connect_layers(0, 1, type='Beam')

    tmp_skin.input_to_load(image, max_force)
    tmp_skin.analyse()

    if vis:
        tmp_skin.visualise()

    displacement = tmp_skin.get_all_displacements()

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
        run_fem(images[-1], layers=2, max_force=1000, mesh_size=ms, vis=True)

    sys.exit()
