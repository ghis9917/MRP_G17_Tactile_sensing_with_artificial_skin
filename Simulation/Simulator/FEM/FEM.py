from PyNite import FEModel3D
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PyNite import Visualization


class Silicone:
    def __init__(self):
        self.E = 0.001
        self.G = 0.0003
        self.Iy = 100 #???
        self.Ix = 100 #???
        self.J = 50 #???
        self.A = 10 #???


silicone = Silicone()
fem = FEModel3D()
nodelist = []
memberlist = []
for i in range(-10, 10):
    for j in range(-10, 10):
        if i == -10 or i == 9 or j == -10 or j == 9:
            name = str(i) + "_" + str(j)
            nodelist.append(name)
            x = i * 10
            z = j * 10
            fem.AddNode(name, x, 0, z)
            fem.DefineSupport(name, True, True, True, True, True, True)

for i in range(len(nodelist)):
    for j in range(i+1, len(nodelist)):
        n = nodelist[i].split("_")
        n2 = nodelist[j].split("_")
        n[0] = int(n[0])
        n[1] = int(n[1])
        n2[0] = int(n2[0])
        n2[1] = int(n2[1])
        if (n[0] == n2[0] and np.abs(n[1] - n2[1]) == 1) or (n[1] == n2[1] and np.abs(n[0] - n2[0]) == 1):
            print("nodes", n, n2)

            fem.AddMember(str(i)+str(j), nodelist[i], nodelist[j],
                          silicone.E, silicone.G, silicone.Ix,
                          silicone.Iy, silicone.J, silicone.A)

fem.Analyze(check_statics=True)

Visualization.RenderModel(fem, text_height=5, deformed_shape=False, deformed_scale=100, render_loads=True)


