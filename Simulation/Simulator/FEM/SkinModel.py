import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

DEFAULT_PLATE = (1, 0.001, 0.17)
DEFAULT_BEAM = (29000, 11400, 100, 150, 250, 20)


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

    def create_nodes(self, width, height, layers=1, mesh_size=1.0, layer_dist=0.5):
        """
        Creates nodes based on the specified width and height of the skin divided by the mesh size

        :param width: Width of the skin sheet
        :param height: Height of the skin sheet
        :param layers: Amount of node layers
        :param mesh_size: Subdivision of the sheet
        :param layer_dist: The distance between each layer
        :return: None
        """
        self.width, self.height = width, height
        self.layers, self.mesh_size = layers, mesh_size

        # Width and height divided by the mesh size to get our node step size
        x_steps = round(np.ceil(width / mesh_size))
        y_steps = round(np.ceil(height / mesh_size))

        print(width, height, width / mesh_size, y_steps, mesh_size)

        # Create a list with each x and y value for the possible nodes
        x_lsp = np.linspace(0, width, x_steps)
        self.xlsp = x_lsp
        y_lsp = np.linspace(0, height, y_steps)
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
                self.fem.AddNode(name, x_cord, y_cord, layer * -layer_dist)
                self.node_name_list.append(name)
                self.node_matrix[x, y, layer] = name

        # Convert the node matrix after creation to a numpy array for easy slicing
        self.node_matrix = np.array(self.node_matrix, dtype=object)

    def create_plates(self, plate_layer=-1, properties=None):
        """
        This method creates plates between each combination of four nodes in the existing node matrix,
        in order to create a 'layer' of the material that we provide for the plates.

        :return: None
        """
        if (self.width is None) | (self.height is None) | (self.layers is None):
            raise ValueError("Tried to add plates without nodes")

        # Get the shape of the node matrix to create a plate matrix
        shap = self.node_matrix.shape
        if self.plate_matrix is None:
            self.plate_matrix = np.array(
                [[["" for _ in range(shap[2])]
                  for _ in range(shap[1] - 1)]
                 for _ in range(shap[0] - 1)],
                dtype=object)

        # If we are doing all layers (plate_layer == -1)
        if plate_layer == -1:
            len_layer = self.layers - 1 if self.layers > 1 else 1
            for layer in range(len_layer):
                self.__create_plates(layer, properties)
        # Else we do one layer
        else:
            self.__create_plates(plate_layer, properties)

    # Extension method of create_plates to prevent duplicate code
    def __create_plates(self, layer, properties):
        for i, j in itertools.product(range(len(self.xlsp) - 1), range(len(self.ylsp) - 1)):
            n1 = 'N' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(layer))
            n2 = 'N' + str(int(i)) + '.' + str(int(j + 1)) + '.' + str(int(layer))
            n3 = 'N' + str(int(i + 1)) + '.' + str(int(j)) + '.' + str(int(layer))
            n4 = 'N' + str(int(i + 1)) + '.' + str(int(j + 1)) + '.' + str(int(layer))

            name = 'P' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(layer))

            if properties is None:
                properties = DEFAULT_PLATE
            self.fem.AddPlate(name, n1, n2, n4, n3, properties[0], properties[1], properties[2])

            self.plate_matrix[i, j, layer] = name

    def connect_layers(self, l0, l1, connection_type='Beam', beam_properties=None, plate_properties=None):
        """
        Connect the different layers of nodes with either a 'Beam' or a 'Plate'

        :param l0: First layer to connect
        :param l1: Second layer to connect
        :param connection_type: Beam or Plate
        :param beam_properties: The material properties of the beam (if that type is used)
        :param plate_properties: The material properties of the plate (if that type is used)
        :return: None
        """
        # Value checks
        if not ((0 <= l0 < self.layers) and (0 <= l1 < self.layers)):
            raise ValueError("Connection cannot be made between non-existent layers")

        # If the connection_type is Beam, we can create a 1-1 mapping between the nodes
        if connection_type == 'Beam':
            layer_0 = self.node_matrix[:, :, l0].ravel()
            layer_1 = self.node_matrix[:, :, l1].ravel()

            for id in range(len(layer_0)):
                n1 = layer_0[id]
                n2 = layer_1[id]
                if beam_properties is None:
                    beam_properties = DEFAULT_BEAM

                self.fem.AddMember(f'M{l0}.{l1}.{id}', n1, n2,
                                   beam_properties[0],
                                   beam_properties[1],
                                   beam_properties[2],
                                   beam_properties[3],
                                   beam_properties[4],
                                   beam_properties[5])

        # Connect each combination of (nearby) nodes with plates
        if connection_type == 'Plate':
            if connection_type == 'Plate':
                raise BrokenPipeError("Right now plate connections are not supported")
            layer_0 = self.node_matrix[:, :, l0]

            visited = []
            one_layer_tuple = []

            for i, j in itertools.product(range(layer_0.shape[0]), range(layer_0.shape[1])):

                others = []
                if i > 0:
                    others.append((i - 1, j))
                if j > 0:
                    others.append((i, j))
                if i < layer_0.shape[0] - 1:
                    others.append((i + 1, j))
                if j < layer_0.shape[1] - 1:
                    others.append((i, j + 1))

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

                plate_name = f'P.{n0}.{n1}.{n2}.{n3}.'

                self.fem.AddPlate(plate_name, n0, n1, n3, n2, self.t, self.E, self.nu)

    def get_node_mat(self):
        return self.node_matrix

    def input_to_load(self, image, max_force, load_type='Plate'):
        """
        Divides the input image in such a way the intensity of the image's values can be translated to pressures
        on nodes or plates

        :param image: The input image with white levels indicating the amount of force
        :param max_force: The image's range [0-255] will be mapped to the force range [0-max_force]
        :param load_type: The type of load, either plate or nodal load
        :return: None
        """
        # Reshape image to mesh size
        dsize = (int(np.ceil(self.width)), int(np.ceil(self.height)))
        image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)

        # Normalize image (test)
        img = image
        if np.sum(image) != 0:
            img = image / np.sum(image)

        if load_type == 'Plate':
            if self.plate_matrix is None:
                raise ValueError('There are no plates to put pressure on!')

        x_step = self.xlsp[1]
        y_step = self.ylsp[1]

        s = self.plate_matrix.shape
        print(s)

        for x, y in itertools.product(range(s[0]), range(s[1])):
            x_0 = int(x * x_step)
            x_1 = int(x * x_step + x_step)
            y_0 = int(y * y_step)
            y_1 = int(y * y_step + y_step)

            window = img[x_0:x_1, y_0:y_1]
            avg = np.mean(window)

            if avg != 0.0:
                force = avg * max_force

                plat_name = self.plate_matrix[x, y, 0]
                case_name = 'Case 1'  # +plat_name[1:]

                if load_type == 'Plate':
                    self.fem.Plates[plat_name].pressures.append([force, case_name])

                elif load_type == 'Node':
                    # TODO shift image to center it if we use node loads
                    self.fem.AddNodeLoad(self.node_matrix[x, y, 0], 'FZ', -force)

    # Obsolte function (DON'T DELETE FOR REFERENCE)
    # def __define_support(self, type="Pinned", loc="Sides"):
    #     """
    #     Sets the nodes on the edges of the skin to be a support, disallowing either movement, rotation or both
    #
    #     :param type: Pinned = no movement, Fixed = no movement, no rotation
    #     :return: None
    #     """
    #     if loc == "Sides":
    #         for l in range(self.layers):
    #             nodes = []
    #             if l < self.layers-1:
    #                 nodes.extend(self.node_matrix[:, 0, l])
    #                 nodes.extend(self.node_matrix[:, -1, l])
    #                 nodes.extend(self.node_matrix[0, :, l])
    #                 nodes.extend(self.node_matrix[-1, :, l])
    #             else:
    #                 nodes = self.node_matrix[:, :, l].ravel()
    #                 print(nodes)
    #
    #             for n in nodes:
    #                 node = self.fem.Nodes[n]
    #                 if type == "Pinned":
    #                     node.SupportDX, node.SupportDY, node.SupportDZ = True, True, True
    #                 elif type == "Fixed":
    #                     node.SupportDX, node.SupportDY, node.SupportDZ, node.SupportRX, node.SupportRY, node.SupportRZ = True, True, True, True, True, True
    #     if loc == "All":
    #         for node in self.fem.Nodes.values():
    #             node.SupportDX, node.SupportDY, node.SupportDZ = True, True, True

    def __set_layer_support(self, layer_nodes, support=('Pinned',), loc='All'):
        """
        Sets the nodes that are passed to a certain support, either fixed along a direction or fully fixed

        :param layer_nodes: List of nodes for a layer
        :param support: Type of support(s) {'Pinned', 'Fixed', 'DX', 'DY', 'DZ', etc.}
        :param loc: Location of nodes that are supported, either 'All' or 'Side'
        :return: None
        """
        support_dict = {
            'Pinned': (True, True, True, False, False, False),
            'Fixed': (True, True, True, True, True, True),
            'DX': (True, False, False, False, False, False),
            'DY': (False, True, False, False, False, False),
            'DZ': (False, False, True, False, False, False),
            'RX': (False, False, False, True, False, False),
            'RY': (False, False, False, False, True, False),
            'RZ': (False, False, False, False, False, True),
            'None': (False, False, False, False, False, False)
        }

        for sup in support:
            if sup not in support_dict.keys():
                print("Support '{sup}' not defined!")
                return

        all_layer_nodes = layer_nodes.ravel()
        if loc == 'All':
            for n in all_layer_nodes:
                node = self.fem.Nodes[n]
                if len(support) > 1:
                    setter = support_dict[support[0]]
                    for sup in support[1:]:
                        setter = np.array(setter) | np.array(support_dict[sup])
                else:
                    setter = support_dict[support[0]]

                node.SupportDX, node.SupportDY, node.SupportDZ, \
                node.SupportRX, node.SupportRY, node.SupportRZ = setter

        elif loc == 'Side':
            pass

    def define_support(self, support_dict=None):
        if support_dict is not None:
            for key, value in support_dict.items():
                self.__set_layer_support(self.node_matrix[:, :, key], support=value)

        else:
            for layer in range(self.layers):
                if self.layers > 1:
                    for layer in range(self.layers):
                        if layer > 0:
                            if layer == self.layers - 1:
                                self.__set_layer_support(self.node_matrix[:, :, layer], support=('Fixed',))
                            else:
                                self.__set_layer_support(self.node_matrix[:, :, layer], support=('DX', 'DY'))

    def get_model(self):
        return self.fem

    def analyse(self):
        # self.__define_support(type='Pinned')

        self.fem.Analyze(check_statics=True, sparse=True, max_iter=30)

    def visualise(self):
        Visualization.RenderModel(self.fem,
                                  text_height=0.1,
                                  deformed_shape=False,
                                  deformed_scale=10,
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

    def get_all_displacements(self, depth=1):
        xsize = self.node_matrix.shape[0]
        ysize = self.node_matrix.shape[1]
        displacement_mat = np.zeros(shape=(xsize, ysize))
        for i, j in itertools.product(range(xsize), range(ysize)):
            t_name = self.node_matrix[i, j, depth]
            tmp_dz = list(self.fem.Nodes[t_name].DZ.values())[0]
            displacement_mat[i, j] = tmp_dz
        return displacement_mat

