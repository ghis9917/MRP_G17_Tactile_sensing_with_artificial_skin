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
import matplotlib.pyplot as plt


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

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 2

class Dataset_from_csv(DGLDataset):
    def __init__(self,
                 list_IDs,
                 window_size=5,
                 stride_frac = 2/3,
                 frames=30,
                 path_adj='../Simulation/out/v6/',
                 path_values='../Simulation/out/v7/'):
        self.list_IDs = list_IDs
        self.window_size = window_size
        self.frames_x_gesture = frames
        self.path_adj = path_adj
        self.path_values = path_values
        self.dim = len(self.list_IDs)
        self.adj = coo_matrix(pd.read_csv(self.path_adj + 'adjacency_matrix.csv').values)
        self.data = pd.read_csv(self.path_values + 'dataset.csv')
        self.graph = dgl.from_scipy(self.adj, 'weight')
        self.sensors_ids = [f'S{i}' for i in range(40)]  # make it automatic
        self.stride_frac = stride_frac #stride fraction

    def __getitem__(self, i):
        id = self.list_IDs[i]
        #print(id)
        graph_list = []
        windows = []
        start = 0
        while True:
            windows.append((start, start + self.window_size))
            start += int(self.stride_frac * self.window_size)
            if start >= self.frames_x_gesture:
                break
            if start + self.window_size > self.frames_x_gesture:
                windows.append((start, self.frames_x_gesture))
                break
        fp = self.data.loc[self.data['id'] == id]
        for start, end in windows:
            # pad last window?
            sensors = torch.Tensor(fp[self.sensors_ids].values[start:end].T)
            g = copy.deepcopy(self.graph)
            g.ndata['feature'] = sensors
            graph_list.append(g)

        labels = torch.Tensor(fp.iloc[0][2:6].values.astype(float))
        return graph_list,  labels

    def __len__(self):
        # Denotes the total number of samples
        return self.dim

class GConvNetSimple(nn.Module):
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

class GConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphConv(10, 8))
        self.output = nn.Linear(8, 4)

    def forward(self, graphs):
        # TODO ghraphs is a list, go over each graph and get embedding for lstm
        return

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

def get_dataloaders_from_csv(window_size=5, stride_frac=2/3):
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
    training_set = Dataset_from_csv(partition['train'], window_size=window_size, stride_frac=stride_frac)
    validation_set = Dataset_from_csv(partition['validation'], window_size=window_size, stride_frac=stride_frac)
    test_set = Dataset_from_csv(ids_test, window_size=window_size, stride_frac=stride_frac)

    return GraphDataLoader(training_set, **params), \
           GraphDataLoader(validation_set, **params), \
           GraphDataLoader(test_set, **params)

class GConvNetBigGraph(nn.Module):
    def __init__(self,
                window_size = 30,
                stride_frac = 1):
        super().__init__()

        self.window_size = window_size
        self.stride_frac = stride_frac

        hidden1 = 30
        hidden2 = 10
        output = 4

        self.conv1 = GraphConv(window_size, hidden1)
        self.hidden = nn.Linear(in_features=hidden1, out_features=hidden2, bias=True)
        self.acthidden = nn.ReLU()
        self.output = nn.Linear(in_features=hidden2, out_features=output, bias=True)
        self.actout = nn.Sigmoid()

    def forward(self, graphs, features):
        #define a NN structure using GDL and Torch layers
        x = self.conv1(graphs, features)
        graphs.ndata['h'] = x
        x = dgl.max_nodes(graphs, 'h')
        x = self.hidden(x)
        x = self.acthidden(x)
        x = self.output(x)
        x = self.actout(x)
        return x

    def train(self, epochs=2, lr=0.01, test_rate=0.8):
        model = self #create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #choose an optimizer
        loss = nn.BCELoss()
        ## Configuring the DataLoader ##
        batch_size = 1
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders_from_csv(window_size=self.window_size,stride_frac=self.stride_frac)
        num_train = train_dataloader.__len__()

        ## Training phase ## 
        acc_history = []
        best_acc = 0
        for epoch in range(epochs):
            print('Epoch ', epoch+1, 'of ', epochs)
            ex = 0 
            for batched_graph, label in train_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph,batched_graph.ndata['feature'].float()) #forward computation on the batched graph
                J = loss(pred, label) #calculate the cost function
                optimizer.zero_grad() #set the gradients to zero
                J.backward()
                optimizer.step() #backpropagate
                print(ex,' / ',num_train)
                ex += batch_size

            #calculate the accuracy on test set and print
            num_correct = 0; num_tests = 0
            for batched_graph, label in test_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph, batched_graph.ndata['feature'].float()) #forward computation on the batched graph
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += (label.shape[0]*label.shape[1])
            acc = num_correct / num_tests
            acc_history.append(acc)
            print('Test of overall accuracy: ', acc)
            if (acc>best_acc):
                best_acc = acc
                self.save()
        
        ## Save the accuracy/epochs report ##
        with open('./logfile.txt','w') as fp:
            fp.write(json.dumps(acc_history))
            print('### Log salvato ###')

        return acc_history

    def evaluation(self):
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders_from_csv(window_size=self.window_size,stride_frac=self.stride_frac)
        print('### Evaluation of the network ###')
        #calculate the accuracy on test set and print
        num_correct = 0; num_tests = 0
        for batched_graph, label in test_dataloader:
            batched_graph = batched_graph[0]
            pred = model(batched_graph, batched_graph.ndata['feature'].float()) #forward computation on the batched graph
            num_correct += ((pred>0.5) == label).sum().item()
            num_tests += (label.shape[0]*label.shape[1])
        print('Evaluation of overall accuracy: ', num_correct / num_tests)

    def save(self, file='GNN_BG.tar'):
        torch.save(self.state_dict(), file)

    def load(file):
        model = GConvNetBigGraph()
        model.load_state_dict(torch.load(file))
        return model


if __name__ == '__main__':
    model = GConvNetBigGraph()
    acc_hist = model.train(epochs=3)
    plt.plot(acc_hist)
    plt.show()
    model_best = GConvNetBigGraph.load('./GNN_BG.tar')
    model_best.evaluation()
    