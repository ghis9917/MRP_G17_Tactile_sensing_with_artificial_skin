import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import json
import dgl
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dataloaders import get_dataloaders_from_graph, get_dataloaders_from_csv


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
    def __init__(self, windows, features):
        super().__init__()
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphConv(features, 8))
        self.GCN_layers.append(GraphConv(8, 16))
        self.temporal_layer = nn.LSTM(16, 8)
        self.output = nn.Linear(8, 4)

    def forward(self, graphs):
        convolutions = []
        for graph in graphs:
            h = graph.ndata['feature'].float()
            for layer in self.GCN_layers:
                h = F.relu(layer(graph, h))
            convolutions.append(h)
        input_features = torch.stack(convolutions)
        out = self.temporal_layer(input_features)[0]
        graph.ndata['feature'] = out[-1]
        x = dgl.max_nodes(graph, 'feature')
        return torch.sigmoid(self.output(x))


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
        print('### Model loaded ###')
        return model


if __name__ == '__main__':
    model = GConvNetBigGraph()
    acc_hist = model.train(epochs=40)
    plt.plot(acc_hist)
    plt.show()
    model_best = GConvNetBigGraph.load('./GNN_BG.tar')
    model_best.evaluation()
