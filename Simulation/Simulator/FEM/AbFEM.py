import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
import cv2
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

    def create_nodes(self, width, height, layers=1, mesh_size=1):
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
        len_layer = self.layers # self.layers - 1 if self.layers > 1 else 1
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



    #TODO Look into padding the nodes of the FEM

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

                case_name = 'Case 1' #+plat_name[1:]
                #self.fem.Plates[plat_name].pressures.append([force, case_name])
                self.fem.AddNodeLoad(self.node_matrix[x, y, 0], 'FZ', -force)

        # for x, y in itertools.product(range(len(self.node_matrix)), range(len(self.node_matrix[0]))):
        #     node_filter = inp[
        #                   x*window:x*window+window,
        #                   y*window:y*window+window
        #                   ]
        #     avg = np.mean(node_filter)
        #     if avg != 0.0:
        #         force = (avg / 255) * max_force
                #self.fem.AddNodeLoad(self.node_matrix[x, y, 0], "FZ", -force)

    def __define_support(self, type="Pinned"):
        """
        Sets the nodes on the edges of the skin to be a support, disallowing either movement, rotation or both
        :param type: Pinned = no movement, Fixed = no movement, no rotation
        :return: None
        """
        for l in range(self.layers):
            nodes = []
            nodes.extend(self.node_matrix[:, 0, l])
            nodes.extend(self.node_matrix[:, -1, l])
            nodes.extend(self.node_matrix[0, :, l])
            nodes.extend(self.node_matrix[-1, :, l])

            for n in nodes:
                node = self.fem.Nodes[n]
                if type == "Pinned":
                    node.SupportDX, node.SupportDY, node.SupportDZ = True, True, True
                elif type == "Fixed":
                    node.SupportDX, node.SupportDY, node.SupportDZ, node.SupportRX, node.SupportRY, node.SupportRZ = True, True, True, True, True, True

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



if __name__ == '__main__':
    skin_model = SkinModel()
    skin_model.create_nodes(100, 100, layers=2, mesh_size=5)
    skin_model.create_plates()

    # skin_model.connect_layers(0, 1)

    # TODO normalize dsize to width & height (??)

    # TODO create method to read in (sequential) images
    inp = cv2.resize(
        cv2.imread('../input/circle blur.png', cv2.IMREAD_GRAYSCALE),
        dsize=(100, 100),
        interpolation=cv2.INTER_CUBIC
    )

    skin_model.input_to_load(inp, 100)
    skin_model.analyse()

    # Plot the shear force contours
    # skin_model.plot_forces('Qx')
    # skin_model.plot_forces('Qy')
    #
    # # Plot the moment contours
    # skin_model.plot_forces('Mx')
    # skin_model.plot_forces('My')
    # skin_model.plot_forces('Mxy')
    skin_model.visualise()

    sys.exit()
