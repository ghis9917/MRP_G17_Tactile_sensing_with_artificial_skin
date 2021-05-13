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

        self.width, self.height = width, height
        self.layers, self.mesh_size = layers, mesh_size

        x_steps = int(width // mesh_size)
        y_steps = int(height // mesh_size)

        x_lsp = np.linspace(0, width, x_steps)
        self.xlsp = x_lsp
        y_lsp = np.linspace(0, height, y_steps)
        self.ylsp = y_lsp

        layer_vec = ["" for _ in range(self.layers)]
        x_mat = [layer_vec for _ in range(len(self.xlsp))]
        self.node_matrix = np.array([x_mat for _ in range(len(self.ylsp))], dtype=object)

        print(self.node_matrix.shape)
        for l in range(layers):
            for x, y in itertools.product(range(len(x_lsp)), range(len(y_lsp))):
                x_cord = x_lsp[x]
                y_cord = y_lsp[y]
                name = 'N' + str(int(x)) + '.' + str(int(y)) + '.' + str(int(l))
                self.fem.AddNode(name, x_cord, y_cord, l*-20)
                self.node_name_list.append(name)
                self.node_matrix[x, y, l] = name

        #self.node_matrix = np.array(self.node_matrix, dtype=object)

    def create_plates(self):
        if (self.width is None) | (self.height is None) | (self.layers is None):
            raise ValueError("Tried to add plates without nodes")

        shap = self.node_matrix.shape
        self.plate_matrix = np.array([[["" for _ in range(shap[2])] for _ in range(shap[1]-1)] for _ in range(shap[0]-1)], dtype=object)
        len_layer = self.layers - 1 if self.layers > 1 else 1
        for l in range(len_layer):
            for i, j in itertools.product(range(len(self.xlsp) - 1), range(len(self.ylsp) - 1)):

                n1 = 'N' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(l))
                n2 = 'N' + str(int(i)) + '.' + str(int(j+1)) + '.' + str(int(l))
                n3 = 'N' + str(int(i+1)) + '.' + str(int(j)) + '.' + str(int(l))
                n4 = 'N' + str(int(i+1)) + '.' + str(int(j+1)) + '.' + str(int(l))

                name = 'P' + str(int(i)) + '.' + str(int(j)) + '.' + str(int(l))

                self.fem.AddPlate(name, n1, n2, n4, n3, self.t, self.E, self.nu)

                self.plate_matrix[i, j, l] = name

    #TODO Look into padding the nodes of the FEM

    def get_node_mat(self):
        return self.node_matrix

    def input_to_load(self, img, max_force):
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
                self.fem.Plates[plat_name].pressures.append([force, case_name])




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
        self.__define_support(type='Fixed')

        self.fem.Analyze(check_statics=True, sparse=True)

    def visualise(self):
        Visualization.RenderModel(self.fem, text_height=0.5, deformed_shape=False, deformed_scale=30,
                                  render_loads=True, color_map='dz', case=None, combo_name='Combo 1')


if __name__ == '__main__':
    skin_model = SkinModel()
    skin_model.create_nodes(100, 100, layers=1, mesh_size=5)
    skin_model.create_plates()

    # TODO normalize dsize to width & height (??)

    # TODO create method to read in (sequential) images
    inp = cv2.resize(
        cv2.imread('../input/hand.png', cv2.IMREAD_GRAYSCALE),
        dsize=(100, 100),
        interpolation=cv2.INTER_CUBIC
    )

    skin_model.input_to_load(inp, 10)
    skin_model.analyse()

    print()

    skin_model.visualise()

    sys.exit()
