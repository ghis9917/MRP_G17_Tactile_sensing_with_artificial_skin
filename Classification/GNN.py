import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl.nn.pytorch.conv import RelGraphConv
import pandas as pd
#from graph_creation import read_graphs, read_graph
import os
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
from torch.utils.data import DataLoader
from dgl.dataloading.pytorch import GraphDataLoader
import json
from GNN_classes import *

## Initial settings ##
SET_SEED=11
# Compatibility with CUDA and GPU -> remember to move into GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#make deterministic the stochastic operation to have better comparable tests
if SET_SEED!=-1:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SET_SEED)
        torch.cuda.manual_seed_all(SET_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(SET_SEED)
    torch.manual_seed(SET_SEED)



## Graph Dataset dynamic loading ##
class MyDataset(DGLDataset):
    def __init__(self, path):
        self.path = path
        super().__init__(name='mydataset')
        self.process()

    def process(self):
        #load pandas dataset
        self.graphs_names = []
        self.labels = []
        self.graph_path = self.path
        #Import from Diego's graphs
        #self.graphs = [dgl.from_networkx(nxg, node_attrs=['pressure_val']) for nxg in read_graphs()]
        labels = np.load('./labels.npy')
        j = 0
        for file in os.listdir(self.graph_path):
            file = file.replace('.json', '')
            if (labels[j]==1):
                self.graphs_names.append(file)
                self.labels.append(torch.tensor([0.0]))
            if (labels[j]==2):
                self.graphs_names.append(file)
                self.labels.append(torch.tensor([1.0]))
            j += 1
        print('Database loaded')

    def __getitem__(self, i):
        graph = dgl.from_networkx(read_graph(self.graphs_names[i]), node_attrs=['pressure_val'])
        return graph, self.labels[i]

    def __len__(self):
        return len(self.graphs_names)

## Graph Neural Network ##
class GCN(nn.Module):
    def __init__(self, in_feat, hidden1, hidden2, output, path='./graphs'):
        super(GCN, self).__init__()
        self.path = path
        #self.conv1 = GraphConv(in_feats, h_feats)
        self.conv1 = GraphConv(in_feat, hidden1)
        self.hidden = nn.Linear(in_features=hidden1, out_features=hidden2, bias=True)
        self.acthidden = nn.ReLU()
        self.output = nn.Linear(in_features=hidden2, out_features=output, bias=True)
        self.actout = nn.Sigmoid()

    def save(self, file='GNN.tar'):
        torch.save(self.state_dict(), file)

    def load(file, in_feat, hidden1, hidden2, output, path='./graphs'):
        model = GCN(in_feat, hidden1, hidden2, output, path)
        model.load_state_dict(torch.load(file))
        return model

    def forward(self, graphs, features):
        #define a NN structure using GDL and Torch layers
        x = self.conv1(graphs, features)
        graphs.ndata['h'] = x
        x = dgl.mean_nodes(graphs, 'h')
        x = self.hidden(x)
        x = self.acthidden(x)
        x = self.output(x)
        x = self.actout(x)
        return x

    def train(self, epochs, lr=0.01, test_rate=0.8):
        model = self #create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #choose an optimizer
        loss = nn.BCELoss()
        ## Configuring the DataLoader ##
        """
        self.dataset = MyDataset(self.path)
        num_examples = self.dataset.__len__()
        num_train = int(num_examples * test_rate)

        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

        batch_size = 20
        
        train_dataloader = GraphDataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
        test_dataloader = GraphDataLoader(self.dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)
        """
        batch_size = 20
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders_from_csv(window_size=30,stride_frac=2)
        num_train = train_dataloader.__len__()

        ## Training phase ## 
        acc_history = []
        for epoch in range(epochs):
            print('Epoch ', epoch+1, 'of ', epochs)
            ex = 0 
            for batched_graph, label in train_dataloader:
                pred = model(batched_graph,batched_graph.ndata['pressure_val'].float().reshape(len(batched_graph.ndata['pressure_val']),1)) #forward computation on the batched graph
                J = loss(pred, label) #calculate the cost function
                optimizer.zero_grad() #set the gradients to zero
                J.backward()
                optimizer.step() #backpropagate
                print(ex,' / ',num_train)
                ex += batch_size

            #calculate the accuracy on test set and print
            num_correct = 0; num_tests = 0
            for batched_graph, label in test_dataloader:
                pred = model(batched_graph, batched_graph.ndata['pressure_val'].float().reshape(len(batched_graph.ndata['pressure_val']),1)) #forward computation on the batched graph
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += (label.shape[0]*label.shape[1])
            acc_history.append(num_correct / num_tests)
            print('Test accuracy: ', num_correct / num_tests)
        
        ## Save the accuracy/epochs report ##
        with open('./logfile.txt','w') as fp:
            fp.write(json.dumps(acc_history))
            print('Log salvato')

    def evaluation(self):
        self.dataset = MyDataset(self.path)
        num_examples = self.dataset.__len__()
        num_train = int(num_examples * 0.5)

        validation_sampler = SubsetRandomSampler(torch.arange(num_train))
        validation_dataloader = GraphDataLoader(self.dataset, sampler=validation_sampler, batch_size=20, drop_last=False)

        with torch.no_grad():
            num_correct = 0; num_tests = 0
            for batched_graph, label in validation_dataloader:
                pred = self(batched_graph, batched_graph.ndata['pressure_val'].float().reshape(len(batched_graph.ndata['pressure_val']),1)) #forward computation on the batched graph
                print(label)
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += (label.shape[0]*label.shape[1])
            print('Test accuracy:', num_correct / num_tests)

if __name__=='__main__':
    #Train
    net = GCN(1, 10, 5, 1, 'D:\MRP2\graphs')
    net.train(2)
    #net.save()
    
    #Evaluate
    #net = GCN.load('./GNN.tar', 1, 10, 5, 1, 'D:\MRP2\graphs')
    net.evaluation()
