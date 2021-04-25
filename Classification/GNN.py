import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import DGLDataset
import pandas as pd

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


## Dataset ##

class MyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='mydataset')

    def process(self):
        #load pandas dataset
        self.graphs = []
        self.labels = []

        #For each graph in the dataset, create a graph structure, save it in the list and save the output label converted as turch tensor, [0,1,0,0] (already one-hot-encoding)
        #g = dgl.graph((src, dst), num_nodes=num_nodes)
        #self.graphs.append(g)
        #self.labels.append(label)
        #self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


## Graph Neural Network ##

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        #self.conv1 = GraphConv(in_feats, h_feats)

    def save(self, file='GNN.tar'):
        torch.save(self.state_dict(), file)

    def load(file='GNN.tar'):
        model = GCN()
        model.load_state_dict(torch_load(file))
        return model

    def forward(self, graphs, in_feat):
        #define a NN structure using GDL and Torch layers
        return True

    def train(self, epochs):
        model = self #create a model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #choose an optimizer

        for epoch in range(epochs):
            for batched_graph, labels in train_dataloader:
                pred = model(batched_graph, batched_graph.ndata['attr'].float()) #batched graphs and input features
                loss = F.cross_entropy(pred, labels) #calculate the cost function
                optimizer.zero_grad() #set the gradients to zero
                loss.backward()
                optimizer.step() #backpropagate

        #Calc accuracy on test set and print
        num_correct = 0
        num_tests = 0
        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata['attr'].float())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        print('Test accuracy:', num_correct / num_tests)

    def evaluation(self):
        self.eval()
        with torch.no_grad():
            num_correct = 0
            num_tests = 0
            for batched_graph, labels in val_dataloader:
                pred = self(batched_graph, batched_graph.ndata['attr'].float())
                num_correct += (pred.argmax(1) == labels).sum().item()
                num_tests += len(labels)
            print('Test accuracy:', num_correct / num_tests)

