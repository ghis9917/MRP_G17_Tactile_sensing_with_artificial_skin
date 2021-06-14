from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
from SkinModel import DEFAULT_PLATE, DEFAULT_BEAM
import matplotlib.pyplot as plt


class ArmModel:
    def __init__(self, layers, diameter=20, height=1, mesh_size=2, dia_dict=None):
        self.fem = FEModel3D()
        self.node_layers = []
        self.center_layer = []
        self.layers = layers
        self.height = height

        self.diameter = diameter
        if dia_dict is None:
            self.dia_dict = {i: (diameter / layers) * (i+1) for i in range(layers)}
        else:
            self.dia_dict = dia_dict
        circum = np.pi * self.diameter  # Calculate the circumference of the cylinder
        self.ring_n = int(np.ceil(circum / mesh_size))    # Based on this the amount of nodes can be calculated
        print(f"Created {self.ring_n} nodes per ring")
        self.grid_scalar = 0

    def create_nodes(self):
        """
        This method creates nodes based on a given mesh size,
        it uses the amount of layers and diameter that that
        were specified in the class initialization

        :param mesh_size: The 'coarseness' of the nodes
        :return: None
        """
        # Create a 'cylinder' of nodes for each layer
        for layer in range(1, self.layers+1):
            diameter = self.dia_dict[layer-1]        # Calculate the diameter of the current ring
            n = self.ring_n
            # Calculate the angle by dividing a full circle by the number of nodes
            steps = (2*np.pi) / n

            # Create a list of node names for each layer apart for easy access later
            node_layer = []

            # Now for the depth of the cylinder we loop over the height
            for h in range(self.height):
                # For the first layer we also create a center
                if layer == 1:
                    c_name = f'N.{h}.c'
                    self.fem.AddNode(c_name, 0, 0, h)
                    self.center_layer.append(c_name)

                # Then for each step in the 'ring' we create a node based on it's geometry
                for c in range(n):
                    x = np.cos(c*steps) * (diameter/2)
                    y = np.sin(c*steps) * (diameter/2)
                    name = f'R.{layer}.{h}.{c}'
                    self.fem.AddNode(name, x, y, h)
                    node_layer.append(name)

            # Then the new layer of nodes is saved
            if layer == 1:
                self.node_layers.append(self.center_layer)
            self.node_layers.append(node_layer)

    def create_plates(self, material_dict=None):
        """
        This method combines all the nodes of one layer with each-other with plates

        :return: None
        """
        if material_dict is None:
            material_dict = {(k+1): DEFAULT_PLATE for k in range(self.layers)}

        # Loop over each layer of nodes to connect them
        for layer in range(1, self.layers+1):
            material = material_dict[layer]
            # Get the names of the nodes in that layer, and calculate the step size based on the length of that list
            layer_nodes = self.node_layers[layer]
            n = int(len(layer_nodes) / self.height)

            # Move over each 'depth' level (except the last)
            for h in range(0, self.height-1):
                for c in range(n):
                    # Retrieve the names based on the naming scheme of the node creation
                    if c == n-1:
                        n0 = f'R.{layer}.{h}.{c}'
                        n1 = f'R.{layer}.{h + 1}.{c}'
                        n2 = f'R.{layer}.{h + 1}.{0}'
                        n3 = f'R.{layer}.{h}.{0}'
                    else:
                        n0 = f'R.{layer}.{h}.{c}'
                        n1 = f'R.{layer}.{h + 1}.{c}'
                        n2 = f'R.{layer}.{h + 1}.{c + 1}'
                        n3 = f'R.{layer}.{h}.{c + 1}'

                    # Create the plate, with the specified material
                    self.fem.AddPlate(f'P.{layer}.{h}.{c}',
                                      n0,
                                      n1,
                                      n2,
                                      n3,
                                      material[0],
                                      material[1],
                                      material[2])

    def connect_layers(self, material_dict=None):
        """
        This method connects each node in a layer with its' corresponding node in the next layer, using a
        membrane

        :return: None
        """
        if material_dict is None:
            material_dict = {k: DEFAULT_BEAM for k in range(self.layers)}

        for layer in range(self.layers):
            material = material_dict[layer]

            for h in range(self.height):
                for c in range(self.ring_n):
                    if layer == 0:
                        n0 = f'N.{h}.c'
                    else:
                        n0 = f'R.{layer}.{h}.{c}'
                    n1 = f'R.{layer+1}.{h}.{c}'
                    name = f'M.{layer}.{h}.{c}'

                    self.fem.AddMember(name, n0, n1,
                                       material[0],
                                       material[1],
                                       material[2],
                                       material[3],
                                       material[4],
                                       material[5]
                                       )

    def create_grid_map(self, scalar=20):
        """
        Rolls out the circular arm onto a template image which can be used to create the plate loads
        using any image editor

        :param scalar: Scalar of the grid, this won't enhance the resolution of the FEM
        :return: Template image
        """
        self.grid_scalar = scalar
        if len(self.fem.Plates) == 0:
            raise ValueError("No plates to create image from")

        height = self.height
        width = self.ring_n

        # TODO add scalar
        output = np.zeros(shape=(height * scalar, width * scalar))

        for xo in range(width):
            x = xo * scalar
            output[:, x] = 255

        for yo in range(height):
            y = yo * scalar
            output[y, :] = 255

        file_name = f'Grid_h{self.height}_ms{self.ring_n}_s{scalar}.jpg'
        plt.imshow(output, cmap='gray')
        plt.savefig('../../input/grids/{}'.format(file_name))
        plt.show()

    # TODO Translate 'panorama' image to projection on plates
    def add_load(self, image_path, max_force):
        image_name = image_path.split('/')[-1]
        image_props = image_name.split('_')[1:]
        image_props[-1] = image_props[-1].split('.')[0]

        if image_props[0] != f'h{self.height}':
            raise Exception(f"Image does not correspond to model - {image_props[0]}")
        elif image_props[1] != f'ms{self.ring_n}':
            raise Exception(f"Image does not correspond to model - {image_props[1]}")
        elif image_props[2] != f's{self.grid_scalar}':
            raise Exception(f"Image does not correspond to model - {image_props[2]}")



    # TODO Add 'fixing' nodes, maybe along non-binary axis is possible?
    def add_support(self):
        pass

    # TODO Get displacement along axis from node to center
    def get_all_displacements(self):
        pass

    def analyse(self):
        self.fem.Analyze(check_statics=True, sparse=True, max_iter=30)

    def visualise(self):
        Visualization.RenderModel(self.fem,
                                  text_height=0.1,
                                  deformed_shape=False,
                                  deformed_scale=300,
                                  render_loads=False
                                  )

    def get_node_layers(self):
        return self.node_layers

    def get_model(self):
        return self.fem


if __name__ == '__main__':
    print("Running ArmModel Directly")
    arm_model = ArmModel(3, height=20, mesh_size=2, dia_dict={0: 2, 1: 10, 2: 12})
    arm_model.create_nodes()
    arm_model.create_plates()
    arm_model.connect_layers()

    arm_model.create_grid_map()
    arm_model.add_load(image_path='../../input/grids/Grid_h20_ms32_s20.jpg', max_force=100)

    inp = input("Analyse? (y/n)")
    if inp.lower() == 'y':
        arm_model.analyse()
        arm_model.visualise()
