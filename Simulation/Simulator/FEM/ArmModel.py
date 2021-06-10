from PyNite import FEModel3D, Node3D
from PyNite import Visualization
import numpy as np
from SkinModel import DEFAULT_PLATE


class ArmModel:
    def __init__(self, layers, diameter=5, height=20):
        self.fem = FEModel3D()
        self.node_layers = []
        self.center_layer = []
        self.layers = layers
        self.height = height
        self.diameter = diameter

    def create_nodes(self, mesh_size=2):
        """
        This method creates nodes based on a given mesh size,
        it uses the amount of layers and diameter that that
        were specified in the class initialization

        :param mesh_size: The 'coarseness' of the nodes
        :return: None
        """
        # Create a 'cylinder' of nodes for each layer
        for layer in range(1, self.layers+1):
            diameter = self.diameter * layer        # Calculate the diameter of the cylinder
            circum = np.pi * diameter               # Calculate the circumference of the cylinder
            n = int(np.ceil(circum / mesh_size))    # Based on this the amount of nodes can be calculated
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

    # TODO Create material_properties dictionary
    def create_plates(self):
        """
        This method combines all the nodes of one layer with each-other with plates

        :return: None
        """
        material = DEFAULT_PLATE
        # Loop over each layer of nodes to connect them
        for layer in range(1, self.layers+1):
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
                        n1 = f'R.{layer}.{h+1}.{c}'
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

    # TODO Write method of beams between layers (& later add material_properties dictionary)
    def connect_layers(self):
        pass

    # TODO Translate 'panorama' image to projection on plates
    def add_load(self, image, max_force):
        pass

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
    arm_model = ArmModel(3)
    arm_model.create_nodes(mesh_size=2)
    arm_model.create_plates()
    arm_model.analyse()
    arm_model.visualise()
