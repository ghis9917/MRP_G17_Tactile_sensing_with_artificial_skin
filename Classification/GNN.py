import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl.nn.pytorch.conv import RelGraphConv
import pandas as pd
from graph_creation import read_graphs, read_graph
import os
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
from torch.utils.data import DataLoader
from dgl.dataloading.pytorch import GraphDataLoader
import json

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
        for file in os.listdir(self.graph_path):
            file = file.replace('.json', '')
            self.graphs_names.append(file)
        #import labels
        self.labels=[torch.tensor([1.0,0.0]),torch.tensor([0.0,1.0])] #Test

    def __getitem__(self, i):
        graph = dgl.from_networkx(read_graph(self.graphs_names[i]), node_attrs=['pressure_val'])
        return graph, self.labels[i]

    def __len__(self):
        return len(self.graphs_names)

    def num_examples(path):
        return len(os.listdir(path))
        
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

    def load(file='GNN.tar'):
        model = GCN()
        model.load_state_dict(torch_load(file))
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
        num_examples = MyDataset.num_examples(self.path)
        num_train = int(num_examples * test_rate)

        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

        self.dataset = MyDataset(self.path)
        train_dataloader = GraphDataLoader(self.dataset, sampler=train_sampler, batch_size=5, drop_last=False)
        test_dataloader = GraphDataLoader(self.dataset, sampler=test_sampler, batch_size=5, drop_last=False)

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

            #calculate the accuracy on test set and print
            num_correct = 0; num_tests = 0
            for batched_graph, label in test_dataloader:
                pred = model(batched_graph, batched_graph.ndata['pressure_val'].float().reshape(len(batched_graph.ndata['pressure_val']),1)) #forward computation on the batched graph
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += label.shape[1]
            acc_history.append(num_correct / num_tests)
            print('Test accuracy: ', num_correct / num_tests)
        
        ## Save the accuracy/epochs report ##
        with open('./logfile.txt','w') as fp:
            fp.write(json.dumps(acc_history))
            print('Log salvato')

    def evaluation(self):
        with torch.no_grad():
            num_correct = 0; num_tests = 0
            for batched_graph, label in val_dataloader:
                pred = self(batched_graph, batched_graph.ndata['pressure_val'].float().reshape(len(batched_graph.ndata['pressure_val']),1)) #forward computation on the batched graph
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += label.shape[1]
            print('Test accuracy:', num_correct / num_tests)


if __name__=='__main__':
    net = GCN(1, 10, 5, 2, './graphs')
    net.train(2)
    net.save()