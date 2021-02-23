import bpy
from scipy.spatial import Delaunay
import numpy as np
from itertools import combinations
 
N_VERTICES = 15

# make mesh
faces = []
edges = []
vertices = [
    [np.random.uniform(0, 5), np.random.uniform(0, 5)] for _ in range(N_VERTICES)
]

# Calculate Delaunay triangulation to get edges and faces of mesh
vertices_for_delaunay = np.array(vertices)
tri = Delaunay(vertices_for_delaunay)
for triangle in tri.simplices:
    edges.extend(set(combinations(triangle, 2)))
    faces.append(triangle)

# Add 3rd dimension to vertices
for v in vertices:
    v.append(0)

new_mesh = bpy.data.meshes.new('new_mesh')
new_mesh.from_pydata(vertices, edges, faces)
new_mesh.update()

# make object from mesh
new_object = bpy.data.objects.new('new_object', new_mesh)

# make collection
new_collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)

# add object to scene collection
new_collection.objects.link(new_object)