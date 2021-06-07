import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx
import json
import ast
from scipy.sparse import coo_matrix, dok_matrix
import dgl
from dgl.data import DGLDataset
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import os.path
from dgl.dataloading import GraphDataLoader
import random
import torch
import copy


class Dataset_from_graphs(DGLDataset):
    def __init__(self,
                 list_IDs,
                 labels,
                 path='graphs_window/',
                 path_attr='graphs_attr_window/'):
        self.labels = labels
        self.list_IDs = list_IDs
        self.path_graphs = path
        self.path_graph_attrs = path_attr
        self.dim = len(self.list_IDs)

    def __getitem__(self, i):
        # generates one sample of data
        # Select sample
        id = self.list_IDs[i]
        print(id)

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

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 2


class Dataset_from_csv(DGLDataset):
    def __init__(self,
                 list_IDs,
                 # labels,
                 path_adj='../Simulation/out/v6/',
                 path_values='../Simulation/out/v7/'):
        # self.labels = labels
        self.list_IDs = list_IDs
        self.graphs_list = []
        self.path_adj = path_adj
        self.path_values = path_values
        self.dim = len(self.list_IDs)
        self.adj = coo_matrix(pd.read_csv(self.path_adj + 'adjacency_matrix.csv').values)
        self.data = pd.read_csv(self.path_values + 'dataset.csv')
        self.graph = dgl.from_scipy(self.adj, 'weight')
        self.sensors_ids = [f'S{i}' for i in range(40)]  # make it automatic

    def __getitem__(self, i):
        id = self.list_IDs[i]
        print(id)
        fp = self.data.loc[self.data['id'] == id]
        sensors = torch.Tensor(fp[self.sensors_ids].values.T)
        g = copy.deepcopy(self.graph)
        g.ndata['feature'] = sensors
        self.graphs_list.append(g)
        labels = fp.iloc[0][2:6].values
        print(labels)
        return self.graphs_list, labels

    def __len__(self):
        # Denotes the total number of samples
        return self.dim


class GConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphConv(10, 8))
        self.GCN_layers.append(GraphConv(8, 4))
        self.output = nn.Linear(4, 1)
        # TODO check k max pooling in pytorch-geometric https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers

    def forward(self, graph, features):
        h = features
        for layer in self.GCN_layers:
            h = F.relu(layer(graph, h))
        graph.ndata['feature'] = h
        graph = dgl.mean_nodes(graph, 'feature')
        # return self.output(graph).view(-1)
        return torch.sigmoid(self.output(graph)).view(-1)


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
    print(statistics)

    # generators
    training_set = Dataset_from_graphs(partition['train'], labels)
    validation_set = Dataset_from_graphs(partition['validation'], labels)
    test_set = Dataset_from_graphs(ids_test, labels)

    return GraphDataLoader(training_set, **params), \
           GraphDataLoader(validation_set, **params), \
           GraphDataLoader(test_set, **params)


def get_dataloaders_from_csv():
    # Parameters
    params = {'batch_size': 1,  # batches merge the nodes in a single graph
              'shuffle': True,
              'num_workers': 1}

    data = pd.read_csv('../Simulation/out/v7/dataset.csv')

    n_gestures = data['id'].iloc[-1]

    # shuffle and divide
    ids = np.arange(n_gestures + 1)
    random.shuffle(ids)
    test_stop = int(len(ids) * 0.8)
    ids_train = ids[:test_stop]
    ids_test = ids[test_stop:]
    val_stop = int(len(ids_train) * 0.8)

    partition = {'train': ids_train[:val_stop], 'validation': ids_train[val_stop:]}

    # generators
    training_set = Dataset_from_csv(partition['train'])
    validation_set = Dataset_from_csv(partition['validation'])
    test_set = Dataset_from_csv(ids_test)

    return GraphDataLoader(training_set, **params), \
           GraphDataLoader(validation_set, **params), \
           GraphDataLoader(test_set, **params)


if __name__ == '__main__':
    training_set, val, test = get_dataloaders_from_csv()
    for c, i in enumerate(training_set):
        print(i)
        exit()
    print('finish', c)
