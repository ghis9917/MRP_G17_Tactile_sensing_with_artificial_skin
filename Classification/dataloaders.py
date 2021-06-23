import os
from math import sqrt
import numpy as np
import pandas as pd
import os.path
from dgl.dataloading import GraphDataLoader
import dgl
import json
import random
import ast
from dgl.data import DGLDataset
import networkx as nx
import copy
from scipy.sparse import coo_matrix
import torch

### OLD APPROACH - Create a dataloader that works creating from scratch a new graph for each example ###
class Dataset_from_graphs(DGLDataset):
    # dataloader that read the graphs from file created with networkx
    def __init__(self,
                 list_IDs,
                 labels,
                 path='graphs_window/', # folder containing graphs
                 path_attr='graphs_attr_window/'): # folder containing the attributes
        self.labels = labels
        self.list_IDs = list_IDs
        self.path_graphs = path
        self.path_graph_attrs = path_attr
        self.dim = len(self.list_IDs)

    def __getitem__(self, i):
        # generates one sample of data
        # Select sample
        id = self.list_IDs[i]
        #print(id)

        # Load data and get label
        with open(self.path_graphs + id, 'r') as js_file_graph:
            graph_nx = nx.from_dict_of_dicts(ast.literal_eval(json.load(js_file_graph)))
        with open(self.path_graph_attrs + id.split('.')[0] + '_attr.json', 'r') as js_file_attr:
            attrs = ast.literal_eval(json.load(js_file_attr))
            attributes_dict = {}
            for entry in attrs:
                attributes_dict[entry[0]] = entry[1]  # TODO make tensors
            nx.set_node_attributes(graph_nx, attributes_dict)

        # create dgl graph
        graph = dgl.from_networkx(graph_nx, node_attrs=['feature'])  # , edge_attrs=['distance'])
        return graph, self.labels[id]

    def __len__(self):
        # Denotes the total number of samples
        return self.dim

def get_dataloaders_from_graph():
    # Parameters
    params = {'batch_size': 1,  # batches merge the nodes in a single graph
              'shuffle': True,
              'num_workers': 1}

    # get graph ids and labels
    ids = os.listdir('graphs_window')[15000:]  # make it balanced
    labels = {}
    labels_dataset = np.load('datasets/labels.npy')
    for i in ids:
        labels[i] = labels_dataset[int(i.split('_')[1])] - 1  # classes loaded 1/2 -> 0/1

    # shuffle and divide
    random.shuffle(ids)
    test_stop = int(len(ids) * 0.8)
    ids_train = ids[:test_stop]
    ids_test = ids[test_stop:]
    val_stop = int(len(ids_train) * 0.8)

    # get labels for train
    partition = {'train': ids_train[:val_stop], 'validation': ids_train[val_stop:]}

    # get statistics
    statistics = {
        'train': {0: 0, 1: 0},
        'val': {0: 0, 1: 0},
        'test': {0: 0, 1: 0}
    }
    for entry in partition['train']:
        statistics['train'][labels[entry]] += 1
    for entry in partition['validation']:
        statistics['val'][labels[entry]] += 1
    for entry in ids_test:
        statistics['test'][labels[entry]] += 1
    #print(statistics)

    # generators
    training_set = Dataset_from_graphs(partition['train'], labels)
    validation_set = Dataset_from_graphs(partition['validation'], labels)
    test_set = Dataset_from_graphs(ids_test, labels)

    return GraphDataLoader(training_set, **params), \
           GraphDataLoader(validation_set, **params), \
           GraphDataLoader(test_set, **params)

### Create a dataloade that works creating one graph and updating the features only ###
class Dataset_from_csv(DGLDataset):
    # dataloader that creates the graphs from a csv file
    # the graphs are generated using an adjacency matrix
    def __init__(self,
                 dataset,
                 list_IDs,
                 window_size=5, # number of frames in the window
                 stride_frac = 2/3,  # fraction of the overlap between a window and the following one
                 test=False, # if true returns the info encoder
                 frames=30, # total number of frames
                 path_adj='', # path of the adjacency matrix
                 path_values=''): # path of the nodes features matrix
        self.list_IDs = list_IDs
        self.window_size = window_size
        self.frames_x_gesture = frames
        self.test = test 
        self.path_adj = path_adj
        self.path_values = path_values
        self.dim = len(self.list_IDs)
        # create adjacency matrix
        self.adj = coo_matrix(pd.read_csv(self.path_adj + 'adjacency_matrix.csv').values) / sqrt(2)
        self.data = dataset
        self.graph = dgl.from_scipy(self.adj, 'weight')
        self.sensors_ids = [f'S{i}' for i in range(self.adj.shape[0])]
        self.stride_frac = stride_frac  # stride fraction
        labels = self.data['shape'].unique()  # unique object shapes
        self.info = {val: i for i, val in enumerate(labels)}  # create object shape encoder string->int

    def get_info(self):
        # we need to encode the shape as number to run the following post processing, in particular the missclassified_obj
        return {key.split('.')[0]: value for key, value in self.info.items()}

    def __getitem__(self, i):
        id = self.list_IDs[i]
        graph_list = []
        windows = []
        start = 0
        # Iterate until possible collecting the start and end indexes of the windows in a gesture
        while True:
            windows.append((start, start + self.window_size))
            start += int(self.stride_frac * self.window_size)
            if start >= self.frames_x_gesture:
                break
            if start + self.window_size > self.frames_x_gesture:
                windows.append((start, self.frames_x_gesture))
                break
        # Cut the gestures into windows and load them into the graph pre-created structure
        # the temporal graph approach will have only one graph, the sequence graph will return a list
        fp = self.data.loc[self.data['id'] == id]
        for start, end in windows:
            sensors = torch.Tensor(fp[self.sensors_ids].values[start:end].T)
            if sensors.shape[1] < self.window_size:
                pad = torch.zeros((40, self.window_size - sensors.shape[1]))
                sensors = torch.cat((sensors, pad), axis=1)
            g = copy.deepcopy(self.graph)
            sensors = torch.nan_to_num(sensors)
            g.ndata['feature'] = (sensors + 2) / (16 + 2) #(sensors + 1) / (15 + 1) value normalization
            graph_list.append(g)

        # return a graph_list, a label_list and the informations about the shape
        labels = torch.Tensor(fp.iloc[0][2:6].values.astype(float))
        info = []
        if self.test:
            pressure_velocity = torch.Tensor(fp.iloc[0][-2:].values.astype(float))
            shape = torch.Tensor([self.info[fp.iloc[0][-3]]])
            info = torch.cat((shape, pressure_velocity))
        return graph_list,  labels, info

    def __len__(self):
        # Denotes the total number of samples
        return self.dim

def check_balance(data, ids):
    # Verify and Calculate the balance level of the dataset #
    statistics = {0: {'label': 'big-small', 0: 0, 1: 0},
                  1: {'label': 'dinamic-static', 0: 0, 1: 0},
                  2: {'label': 'press-tab', 0: 0, 1: 0},
                  3: {'label': 'dangerous-safe', 0: 0, 1: 0}} # statistical information to verify the balance level of our test set
    for val in ids:
        label = data.loc[data['id'] == val].values[0, 2:6]
        for i, val in enumerate(label):
            statistics[i][val] += 1
    return [val for val in statistics.values()]

def get_dataloaders_from_csv(window_size=5, stride_frac=2/3):
    # Parameters
    params = {'batch_size': 1,  # batches merge the nodes in a single graph
              'shuffle': True,
              'num_workers': 0}

    data = pd.read_csv('results_fem_18.csv') # dataset path

    #get the unique gestures ids
    ids = data['id'].unique()

    # shuffle them and divide into three sets
    random.shuffle(ids)
    test_stop = int(len(ids) * 0.9)
    ids_train = ids[:test_stop]
    ids_test = ids[test_stop:]
    val_stop = int(len(ids_train) * 0.8)

    partition = {'train': ids_train[:val_stop], 'validation': ids_train[val_stop:]}

    #check the balancing level of the dataset
    print('training data: ', check_balance(data, ids_train))
    print('testing data: ', check_balance(data, ids_test))

    # generate the datasets
    training_set = Dataset_from_csv(data, partition['train'], window_size=window_size, stride_frac=stride_frac)
    validation_set = Dataset_from_csv(data, partition['validation'],  window_size=window_size, stride_frac=stride_frac)
    test_set = Dataset_from_csv(data, ids_test, window_size=window_size, stride_frac=stride_frac, test=True)
    
    # return the dataloaders based on the datasets 
    return GraphDataLoader(training_set, **params), \
           GraphDataLoader(validation_set, **params), \
           GraphDataLoader(test_set, **params), test_set.get_info()


if __name__=='__main__':
    train_dataloader, \
        validation_dataloader, \
        test_dataloader, \
        info_encoder = get_dataloaders_from_csv(window_size=30, stride_frac=1)

    for g, l, _ in train_dataloader:
        print(g,l,_)
