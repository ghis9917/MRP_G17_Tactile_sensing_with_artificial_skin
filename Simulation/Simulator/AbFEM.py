import itertools
import sys
from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np

class SkinModel:
    def __init__(self):
        self.fem = FEModel3D()
        self.t = 1
        self.E = 3824
        self.nu = 0.3

        self.width = None
        self.height = None
        self.layers = None
        self.xlsp = None
        self.ylsp = None

        self.node_name_list = []

    def create_nodes(self, width, height, layers=1, mesh_size=1):

        self.width, self.height, self.layers = width, height, layers

        x_steps = int(width // mesh_size)
        y_steps = int(height // mesh_size)

        x_lsp = np.linspace(0, width, x_steps)
        self.xlsp = x_lsp
        y_lsp = np.linspace(0, height, y_steps)
        self.ylsp = y_lsp

        for l in range(layers):
            for x, y in itertools.product(x_lsp, y_lsp):
                name = 'N' + str(int(x)) + '.' + str(int(y)) + '.' + str(int(l))
                self.fem.AddNode(name, x, y, l*20)
                self.node_name_list.append(name)

    def create_plates(self):
        if (self.width is None) | (self.height is None) | (self.layers is None):
            raise ValueError("Tried to add plates without nodes")

        for l in range(self.layers):
            for x, y in itertools.product(self.xlsp, self.ylsp):
                pass

    def get_model(self):
        return self.fem

    def analyse(self):
        self.fem.Analyze(check_statics=True, sparse=True)

    def visualise(self):
        Visualization.RenderModel(self.fem, text_height=1)




if __name__ == '__main__':
    skin_model = SkinModel()
    skin_model.create_nodes(100, 100, layers=2, mesh_size=10)
    skin_model.analyse()
    skin_model.visualise()

    sys.exit()

