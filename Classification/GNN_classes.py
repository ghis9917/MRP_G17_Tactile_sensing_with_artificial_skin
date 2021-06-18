import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import json
import dgl
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dataloaders import get_dataloaders_from_graph, get_dataloaders_from_csv
import seaborn as sns
from sklearn.metrics import confusion_matrix
from GNN_frames import visualize_loss
from prettytable import PrettyTable
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

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


def matrix_confusion_overall(y_test, y_pred, model='Model'):
    # create labels for matrix
    labels = unique_labels(y_test, y_pred)
    # create confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels)
    sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
    # set title and labels
    plt.title('Confusion matrix ' + model)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.savefig('confusion_matrix_' + model + '.jpg')
    plt.show()

def matrix_confusion(y_test, y_pred, title='Model'):
    labels = ['big/small', 'dynamic/static', 'press/tap', 'dangerous/safe']
    # create confusion matrix
    matrix = multilabel_confusion_matrix(y_test, y_pred)
    for i in range(4):
        print(matrix[i])
        sns.heatmap(matrix[i], square=True, annot=True, fmt='d', cbar=False)
        # set title and labels
        plt.title('Confusion Matrix_' + labels[i])
        plt.ylabel('true label')
        plt.xlabel('predicted label')
        # plt.savefig('Plots/Classification/confusion_matrix_' + title + '.png')
        plt.show()

def class_to_label(y):
    y = y.type(torch.uint8)
    classes = [['Big','Small'],['Dynamic','Static'],['Press','Tap'],['Dangeours','Safe']]
    return f'{classes[0][y[0,0].item()]}/{classes[1][y[0,1].item()]}/{classes[2][y[0,2].item()]}/{classes[3][y[0,3].item()]}'

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

class GConvNetFrames(nn.Module):
    def __init__(self, device, window_size=30, stride_frac=1):
        super().__init__()
        self.window_size = window_size
        self.stride_frac = stride_frac

        self.activation = nn.SiLU()

        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphConv(1, 64, activation=self.activation))
        self.GCN_layers.append(GraphConv(64, 256, activation=self.activation))
        self.GCN_layers.append(GraphConv(256, 512, activation=self.activation))

        self.GCN_layers_norm = nn.ModuleList()
        self.GCN_layers_norm.append(nn.LayerNorm(64))
        self.GCN_layers_norm.append(nn.LayerNorm(256))
        self.GCN_layers_norm.append(nn.LayerNorm(512))

        # self.pooling = MaxPooling()
        # self.temporal_layer = nn.GRU(512, 32, bidirectional=True)

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(512, 128))
        self.dense_layers.append(nn.LayerNorm(128))
        self.dense_layers.append(nn.Linear(128, 64))
        self.dense_layers.append(nn.LayerNorm(64))
        self.dense_layers.append(nn.Linear(64, 16))
        self.dense_layers.append(nn.LayerNorm(16))
        self.dense_layers.append(nn.Linear(16, 4))

        self.device = device
        self.to(device)
        self.count_parameters()

    def count_parameters(self):
        model = self
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def forward_v0(self): #, graphs):
        convolutions = []
        for graph in graphs:
            h = graph.ndata['feature'].float()
            for layer, norm in zip(self.GCN_layers, self.GCN_layers_norm):
                h = layer(graph, h)
                h = norm(h)
            graph.ndata['feature'] = h
            convolutions.append(self.pooling(graph, h))
        input_features = torch.stack(convolutions)
        # temporal layer
        out, _ = torch.max(input_features, 0)
        # out = self.temporal_layer(input_features)[0]
        # last_forward = out[int(out.shape[0] / 2) - 1, :, :]
        # last_backward = out[-1, :, :]
        # # concat
        # out = torch.cat((last_forward, last_backward), dim=1)
        for dense in self.dense_layers:
            out = self.activation(dense(out))
        return torch.sigmoid(torch.reshape(out, (1, 4)))

    def forward(self, graphs):
        convolutions = []
        for graph in graphs:
            feat = graph.ndata['feature'].float()
            for i in range(feat.shape[-1]):
                h = torch.unsqueeze(feat[:, i], 1)
                for layer, norm in zip(self.GCN_layers, self.GCN_layers_norm):
                    h = layer(graph, h)
                    h = norm(h)
                convolutions.append(h)
        input_features = torch.stack(convolutions)
        max_frame, _ = torch.max(input_features, 0)  # max temporal pooling
        out, _ = torch.max(max_frame, 0)  # max spacial pooling
        for dense in self.dense_layers:
            out = self.activation(dense(out))
        return torch.sigmoid(torch.reshape(out, (1, 4)))

    def train_loop(self, train_dataloader, validation_dataloader, epochs=25, lr=0.001):
        model = self  # create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # choose an optimizer
        loss = nn.BCELoss()
        ## Configuring the DataLoader ##
        batch_size = 1
        num_train = train_dataloader.__len__()

        ## Training phase ##
        history = {
            'loss': [],
            'val_loss': []
        }
        acc_history = []
        best_acc = 0
        for epoch in range(epochs):
            print('Epoch ', epoch + 1, 'of ', epochs)
            ex = 0
            loss_epoch = 0.0

            model.train()

            for batched_graph, label in train_dataloader:
                batched_graph = [graph.to(self.device) for graph in batched_graph]
                label = label.to(self.device)
                pred = model(batched_graph)  # forward computation on the batched graph
                J = loss(pred, label)  # calculate the cost function
                optimizer.zero_grad()  # set the gradients to zero
                J.backward()
                optimizer.step()  # backpropagate
                loss_epoch += J.detach().item()
                if ex % 200 == 0:
                    print(ex, ' / ', num_train)
                ex += batch_size
            # calculate the loss
            epoch_loss = loss_epoch / num_train
            print(f"Epoch {epoch + 1}, loss: {epoch_loss}")
            history['loss'].append(epoch_loss)

            model.eval()
            model.zero_grad()

            # calculate the accuracy on test set and print
            num_correct = 0
            num_tests = 0
            val_loss = 0
            for batched_graph, label in validation_dataloader:
                batched_graph = [graph.to(self.device) for graph in batched_graph]
                label = label.to(self.device)
                pred = model(batched_graph)  # forward computation on the batched graph
                J = loss(pred, label)
                val_loss += J.detach().item()
                num_correct += ((pred > 0.5) == label).sum().item()
                num_tests += (label.shape[0] * label.shape[1])
            acc = num_correct / num_tests
            acc_history.append(acc)
            val_loss = val_loss / len(validation_dataloader)
            print(f"Val Loss for epoch {epoch + 1}: {val_loss}")
            history['val_loss'].append(val_loss)
            print('Test of overall accuracy: ', acc)
            if (acc > best_acc):
                best_acc = acc
                self.save()

        ## Save the accuracy/epochs report ##
        with open('./logfile_GNNframes.txt', 'w') as fp:
            fp.write(json.dumps(acc_history))
            print('### Log salvato ###')

        visualize_loss(history)

        return acc_history

    def evaluation(self, test_dataloader):
        model = self
        print('### Evaluation of the network ###')
        # calculate the accuracy on test set and print
        num_correct = 0;
        num_tests = 0
        for batched_graph, label in test_dataloader:
            batched_graph = [graph.to(self.device) for graph in batched_graph]
            label = label.to(self.device)
            pred = model(batched_graph)  # forward computation on the batched graph
            num_correct += ((pred > 0.5) == label).sum().item()
            num_tests += (label.shape[0] * label.shape[1])
        print('Evaluation of overall accuracy: ', num_correct / num_tests)

    def save(self, file='GNN_frames.tar'):
        torch.save(self.state_dict(), file)

    def load(file):
        model = GConvNetFrames(device)
        model.load_state_dict(torch.load(file))
        print('### Model loaded ###')
        return model

class GConvNetBigGraph(nn.Module):
    def __init__(self,
                window_size = 30,
                stride_frac = 1):
        super().__init__()

        self.window_size = window_size
        self.stride_frac = stride_frac

        hidden1 = 500
        hidden2 = 20
        output = 4

        self.loss = nn.BCELoss()

        self.conv1 = GraphConv(window_size, 250, bias=True, activation=nn.SiLU())
        self.conv2 = GraphConv(250, hidden1, bias=True, activation=nn.SiLU())
        self.hidden = nn.Linear(in_features=hidden1, out_features=hidden2, bias=True)
        self.acthidden = nn.SiLU()
        self.output = nn.Linear(in_features=hidden2, out_features=output, bias=True)
        self.actout = nn.Sigmoid()

        self.count_parameters()

    def forward(self, graphs, features):
        #define a NN structure using GDL and Torch layers
        x = self.conv1(graphs, features)
        x = self.conv2(graphs, x) 
        #x = dgl.nn.SetTransformerEncoder(30, 4, 4, 20, dropouth = 0.9, dropouta=0.9)(graphs, features)
        graphs.ndata['h'] = x
        x = dgl.nn.MaxPooling()(graphs, x) #dgl.nn.WeightAndSum(500)(graphs, x)#
        x = self.hidden(x)
        x = self.acthidden(x)
        x = nn.Dropout(p=0.2)(x)
        x = self.output(x)
        x = self.actout(x)
        return x

    def count_parameters(self):
        model = self
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def train(self, train_dataloader, validation_dataloader, epochs=70, lr=0.001, test_rate=0.8):
        model = self #create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #choose an optimizer
        ## Configuring the DataLoader ##
        batch_size = 1
        num_train = train_dataloader.__len__()

        ## Training phase ##
        acc_history = []
        best_acc = 0
        for epoch in range(epochs):
            print('Epoch ', epoch+1, 'of ', epochs)
            ex = 0
            running_loss = 0.0
            epoch_loss = 0.0
            for batched_graph, label in train_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph,batched_graph.ndata['feature'].float()) #forward computation on the batched graph
                J = self.loss(pred, label) #calculate the cost function
                optimizer.zero_grad() #set the gradients to zero
                J.backward()
                optimizer.step() #backpropagate
                running_loss += J.detach().item()
                ex += batch_size
                if ex % 200 == 0:
                    print(f'{epoch}/{ex+1} -> Loss: {running_loss/200}')
                    epoch_loss += running_loss
                    running_loss = 0.0
            #calculate the accuracy on test set and print
            epoch_loss = epoch_loss/num_train
            print(f'Epoch Loss: {epoch_loss}')
            num_correct = 0.0; num_tests = 0.0; val_loss = 0.0
            for batched_graph, label in test_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph, batched_graph.ndata['feature'].float()) #forward computation on the batched graph
                J = self.loss(pred, label)
                val_loss += J.detach().item()
                num_correct += ((pred>0.5) == label).sum().item()
                num_tests += (label.shape[0]*label.shape[1])
            acc = num_correct / num_tests
            acc_history.append(acc)
            print(f'Test of overall Accuracy: {acc}, Validation Loss: {val_loss}')
            if (acc>best_acc):
                best_acc = acc
                self.save()

        ## Save the accuracy/epochs report ##
        with open('./logfile.txt','w') as fp:
            fp.write(json.dumps(acc_history))
            print('### Log saved ###')

        return acc_history

    def evaluation(self, test_dataloader):
        model = self
        print('### Evaluation of the network ###')
        #calculate the accuracy on test set and print
        num_correct_class=torch.zeros((1,4)); num_test_class=torch.zeros((1,4))
        num_correct = 0.0; num_tests = 0.0; test_loss = 0.0
        y_pred =[]; y_test=[]
        for batched_graph, label in test_dataloader:
            batched_graph = batched_graph[0]
            pred = model(batched_graph, batched_graph.ndata['feature'].float()) #forward computation on the batched graph
            # Accuracy and Loss
            J = self.loss(pred, label)
            test_loss += J.detach().item()
            pred = pred>0.5
            num_correct += (pred == label).sum().item()
            num_tests += (label.shape[0]*label.shape[1])

            # Confusion matrix
            y_pred.append(class_to_label(pred>0.5))
            y_test.append(class_to_label(label))

            # Per Clcass accuracy
            num_correct_class += (pred == label)
            num_test_class += 1

        acc = num_correct / num_tests
        print(f'Overall accuracy: {acc}, Loss: {test_loss}')

        num_correct_class = num_correct_class/num_test_class
        classes = ['Big/Small','Dynamic/Static','Press/Tap','Dangeours/Safe']
        for i in range(4):
            print(f'{classes[i]} -> Accuracy: {num_correct_class[0,i]}')

        matrix_confusion(y_test, y_pred, 'Big Graph')


    def save(self, file='GNN_BG.tar'):
        torch.save(self.state_dict(), file)

    def load(file):
        model = GConvNetBigGraph()
        model.load_state_dict(torch.load(file))
        print('### Model loaded ###')
        return model


if __name__ == '__main__':
    model = GConvNetBigGraph(30,1)
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders_from_csv(window_size=30,stride_frac=1)
    acc_hist = model.train(train_dataloader, validation_dataloader, epochs=70)
    plt.plot(acc_hist)
    plt.show()
    model_best = GConvNetBigGraph.load('./GNN_BG.tar')
    model_best.evaluation(test_dataloader)


    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GConvNetFrames(device)

    train_dataloader, \
        validation_dataloader, \
        test_dataloader = get_dataloaders_from_csv(window_size=model.window_size, stride_frac=model.stride_frac)

    acc_hist = model.train_loop(train_dataloader, validation_dataloader)
    plt.plot(acc_hist)
    plt.show()
    model_best = GConvNetFrames.load('./GNN_frames.tar')
   """
