import ast
import json
import os
import networkx as nx
import numpy as np
from embeddings import *


'''
how to connect the nodes?
how to connect graph of different frame?
'''


def display_graphs_properties(graph, mode=0):
    """
    :param mode: set the values to dispay:
        0: no display
        1: display number of nodes/edges
        2: data og the graph
        3: plot graph
    """

    if mode > 0:
        #print('connected: ', nx.is_connected(graph))
        print('nodes: ', graph.number_of_nodes())
        print('edges: ', graph.number_of_edges())

    if mode > 1:
        print(graph.nodes)
        print(graph.nodes.data())
        print(graph.edges)
        print(graph.adj)

    if mode > 2:
        nx.draw(graph, pos=nx.spring_layout(graph), node_size=30)
        plt.show()


def get_undirected_edges(index, start_value):
    end_value = start_value + 8
    edges = []
    for i in list((index[0] - 1, index[0], index[0] + 1)):
        if start_value <= i < end_value:
            for j in list ((index[1]-1, index[1], index[1]+1)):
                if 0 <= j < 8 and (i, j) != (index[0], index[1]):
                    edges.append([(index[0] * 8) + index[1], (i * 8) + j])
    return edges


def create_whole_graph(sensor_values, count, mode=0):
    print(f'creation of whole graph number {count}...\nframes in sequence:', len(sensor_values))
    graph = nx.DiGraph()
    n_sensors = sensor_values[0].shape[0] * sensor_values[0].shape[1]
    for count, frame in enumerate(sensor_values):
        start_id = count * n_sensors
        for (x, y), value in np.ndenumerate(frame):
            id = start_id + ((x * 8) + y)
            index_new = ((count * 8) + x, y)
            graph.add_node(id, pressure_val=value)
            edges = get_undirected_edges(index_new, count * 8)
            graph.add_edges_from(edges, features={'distance': 1, 'time': 0})
        if start_id != 0:
            graph.add_edge(start_id - 1, start_id, features={'distance': 0, 'time': 1})

    display_graphs_properties(graph, mode)
    return graph


def create_frame_graph(sensor_values, count, mode=0):
    graph = nx.Graph()
    if count % 100 == 0:
        print(f'creation of frame graphs from set number {count}...'
              f'\nnumber of values', len(sensor_values))
    graph_list = []
    for frame in sensor_values:
        for (x, y), value in np.ndenumerate(frame):
            id = (x * 8) + y
            graph.add_node(id, feature=value)
            graph.add_edges_from(get_undirected_edges((x, y), 0), distance=1)

        display_graphs_properties(graph, mode)

        graph_list.append(graph)
    return graph_list


def save_graph(graph, count):
    # create a new directory if it doesn't exist yet
    ROOT_DIR = os.path.abspath(os.curdir)
    path = ROOT_DIR + '/graphs'
    path1 = ROOT_DIR + '/graphs_attr'
    try:
        os.mkdir(path)
    except OSError:
        pass
    try:
        os.mkdir(path1)
    except OSError:
        pass
    with open(f'graphs/graph{count}.json', 'w') as f:
        json.dump(str(nx.to_dict_of_dicts(graph)), f)
    with open(f'graphs_attr/graph{count}_attr.json', 'w') as f:
        json.dump(str(graph.nodes.data()), f)


def read_graphs():
    graphs = []
    for file in os.listdir('graphs'):
        with open(f'graphs/{file}', 'r') as js_file_graph:
            graphs.append(nx.from_dict_of_dicts(ast.literal_eval(json.load(js_file_graph))))

    for count, file in enumerate(os.listdir('graphs_attr')):
        with open(f'graphs_attr/{file}', 'r') as js_file_attr:
            attrs = ast.literal_eval(json.load(js_file_attr))
            attributes_dict = {}
            for entry in attrs:
                attributes_dict[entry[0]] = entry[1]
            nx.set_node_attributes(graphs[count], attributes_dict)
    return graphs


def graph_creation(whole=True):
    """
        :param whole: set the graph to create:
            True: create entire graph with all frames
            False: create a single graph for each frame
        """
    graphs = []
    try:
        training_frames = np.load('datasets/frames.npy', allow_pickle=True)
    except FileNotFoundError:
        print('Not possible to load data. Directory "datasets" does not exist. '
              '\nPlease create directory and add "frames.npy"')
        exit()
    for count, entry in enumerate(training_frames):
        if count < 10: #TODO REMOVEEEEEEEEEEE later
            if whole:
                save_graph(create_whole_graph(entry, count), count)
            else:
                graphs.append(create_frame_graph(entry, count))
    if not whole:
        get_embeddings(graphs)


if __name__ == "__main__":
    #graph_creation(whole=True)
    #graph_creation(whole=False)



