from PyNite import FEModel3D
import numpy as np
import scipy
import matplotlib.pyplot as plt


fem = FEModel3D()

for i in range(-10, 9):
    for j in range(-10, 9):
        name = "n"+str(i)+str(j)
        x = i*10
        y = j*10
        fem.AddNode(name, x, y, 0)
        if i == -10 or i == 9 or j == -10 or j == 9:

            fem.DefineSupport(name, True, True, True, True, True, True)

        