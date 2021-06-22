import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, Set2Set, GlobalAttentionPooling
import json
import dgl
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dataloaders import get_dataloaders_from_graph, get_dataloaders_from_csv
import seaborn as sns
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

"""
## Initial settings ##
SET_SEED = 69
# Compatibility with CUDA and GPU -> remember to move into GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# make deterministic the stochastic operation to have better comparable tests
if SET_SEED != -1:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SET_SEED)
        torch.cuda.manual_seed_all(SET_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(SET_SEED)
    torch.manual_seed(SET_SEED)
"""

def visualize_loss(history, model='Model'):
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = list(range(len(loss)))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(f'loss visualization')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f'images/loss_{model}.jpg')
    plt.close()
    # plt.show()


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
    plt.savefig('images/full_confusion_matrix_' + model + '.jpg')
    plt.close()
    # plt.show()


def matrix_confusion(y_test, y_pred, model='Model'):
    labels = ['big-small', 'dynamic-static', 'press-tap', 'dangerous-safe']
    # create confusion matrix
    matrix = multilabel_confusion_matrix(y_test, y_pred)
    for i in range(4):
        print(matrix[i])
        sns.heatmap(matrix[i], square=True, annot=True, fmt='d', cbar=False)
        # set title and labels
        plt.title('Confusion Matrix_' + labels[i])
        plt.ylabel('true label')
        plt.xlabel('predicted label')
        plt.savefig(f'images/confusion_matrix_{labels[i]}_{model}.png')
        plt.close()
        # plt.show()


def class_to_label(y):
    y = y.type(torch.uint8)
    classes = [['Big', 'Small'], ['Dynamic', 'Static'], ['Press', 'Tap'], ['Dangeours', 'Safe']]
    #return f'{classes[0][y[0, 0].item()]}/{classes[1][y[0, 1].item()]}/{classes[2][y[0, 2].item()]}/{classes[3][y[0, 3].item()]}'
    return [ classes[0][y[0, 0].item()] , classes[1][y[0, 1].item()], classes[2][y[0, 2].item()], classes[3][y[0, 3].item()] ]

def missclassified_obj(statistics, info_encoder, model='Model'):
    list_shapes = list(info_encoder.values())
    max_pressure = 1300
    discretization = 13
    max_velocity = 14
    for c in statistics:
        t = np.squeeze(torch.stack(statistics[c]['obj info']).numpy())
        # binarize shapes and show bars
        bin_obj_shape = np.bincount(t[:, 0].astype(int))
        if bin_obj_shape.size < len(list_shapes):
            bin_obj_shape = np.pad(bin_obj_shape, (0, len(list_shapes) - bin_obj_shape.size), 'constant', constant_values=0)
        plt.bar(list_shapes, bin_obj_shape)
        plt.xticks(list_shapes, list(info_encoder.keys()))
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to object shape')
        plt.savefig(f'images/statistics_{statistics[c]["class"]}_shape_{model}.png')
        plt.close()
        # binarize pressure values and show bars
        bin_pressure = np.bincount(t[:, 1].astype(int))
        if bin_pressure.size < max_pressure:
            bin_pressure = np.pad(bin_pressure, (0, max_pressure - bin_pressure.size), 'constant', constant_values=0)
        # discretize bins
        bin_pressure = np.asarray([np.sum(chunk) for chunk in np.split(bin_pressure, discretization)])
        plt.bar(np.arange(discretization), bin_pressure)
        plt.xticks(np.arange(discretization), np.arange(0, max_pressure, max_pressure / discretization))
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to pressure')
        plt.savefig(f'images/statistics_{statistics[c]["class"]}_pressure_{model}.png')
        plt.close()
        # binarize velocity and show bars
        bin_velocity = np.bincount((t[:, -1] * 10).astype(int))
        if bin_velocity.size < max_velocity:
            bin_velocity = np.pad(bin_velocity, (0, max_velocity - bin_velocity.size), 'constant', constant_values=0)
        plt.bar(np.arange(max_velocity), bin_velocity)
        plt.xticks(np.arange(max_velocity), np.arange(max_velocity) / 10)
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to velocity')
        plt.savefig(f'images/statistics_{statistics[c]["class"]}_velocity_{model}.png')
        plt.close()


class GConvNetFrames(nn.Module):
    def __init__(self, device, window_size=1, stride_frac=1):
        super().__init__()
        self.name = 'sequence graph'
        self.window_size = window_size
        self.stride_frac = stride_frac

        self.loss = nn.BCELoss()

        self.activation = nn.SiLU()

        hidden = 128

        # GConv layers
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphConv(1, 32, activation=self.activation))
        self.GCN_layers.append(GraphConv(32, hidden, activation=self.activation))

        # self.GCN_layers_norm = nn.ModuleList()
        # self.GCN_layers_norm.append(nn.LayerNorm(32))
        # self.GCN_layers_norm.append(nn.LayerNorm(hidden))

        # temporal layer
        self.temporal_layer = nn.GRU(hidden, hidden, bidirectional=False)

        # self.conv = nn.Conv1d(128, 256, kernel_size=5)
        # self.p = nn.MaxPool1d(4)
        # self.conv1 = nn.Conv1d(256, 512, kernel_size=7)

        # pooling
        # self.pooling = MaxPooling()
        # self.s2s = Set2Set(128, 1, 1)
        self.att_pool = GlobalAttentionPooling(nn.Linear(hidden, 1))

        # dense layers
        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(hidden, 16))
        # self.dense_layers.append(nn.GroupNorm(16, 16))
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

    def forward(self, graphs):
        try:
            features = []
            for graph in graphs:
                h = graph.ndata['feature'].float()
                # for layer, norm in zip(self.GCN_layers, self.GCN_layers_norm):
                for layer in self.GCN_layers:
                    h = layer(graph, h)
                    # h = norm(h)
                graph.ndata['feature'] = h
                features.append(h)
            batch_graphs = dgl.batch(graphs)
            batch_f = torch.cat(features, 0)
            # out = self.s2s(batch_graphs, batch_f)
            out = self.att_pool(batch_graphs, batch_f)
            _, out = self.temporal_layer(torch.reshape(out, (len(graphs), 1, out.shape[-1])))
            for dense in self.dense_layers:
                out = self.activation(dense(torch.reshape(out, (1, out.shape[-1]))))
            return torch.sigmoid(torch.reshape(out, (1, 4)))
        except RuntimeError as e:
            print('sample skipped: ', e)
            print(graphs, [graph.ndata for graph in graphs])
            return None

    def train_loop(self, train_dataloader, validation_dataloader, epochs=50, lr=0.001):
        model = self  # create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # choose an optimizer
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
            skipped = 0
            running_loss = 0.0
            epoch_loss = 0.0

            model.train()

            for graphs, label, _ in train_dataloader:
                batched_graph = [graph.to(self.device) for graph in graphs]
                label = label.to(self.device)
                pred = model(batched_graph)  # forward computation on the batched graph
                if pred is not None:
                    J = self.loss(pred, label)  # calculate the cost function
                    optimizer.zero_grad()  # set the gradients to zero
                    J.backward()
                    optimizer.step()  # backpropagate
                    running_loss += J.detach().item()
                    ex += batch_size
                    if ex % 300 == 0:
                        print(f'{epoch + 1}/{ex + 1}-{num_train} -> Loss: {running_loss / 300}')
                        epoch_loss += running_loss
                        running_loss = 0.0
                else:
                    skipped += 1

            # calculate the loss
            epoch_loss = epoch_loss / (num_train - skipped)
            print(f"Epoch {epoch + 1}, loss: {epoch_loss}")
            history['loss'].append(epoch_loss)

            model.eval()
            model.zero_grad()

            # calculate the accuracy on test set and print
            num_correct = 0
            num_tests = 0
            val_loss = 0
            for batched_graph, label, _ in validation_dataloader:
                batched_graph = [graph.to(self.device) for graph in batched_graph]
                label = label.to(self.device)
                pred = model(batched_graph)  # forward computation on the batched graph
                if pred is not None:
                    J = self.loss(pred, label)
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
            print('### Log saved ###')

        visualize_loss(history, model.name)

        return acc_history

    def evaluation(self, test_dataloader, info_encoder):
        model = self
        print('### Evaluation of the network ###')
        # calculate the accuracy on test set and print
        statistics = {0: {'class': 'big-small', 'obj info': []},
                      1: {'class': 'dynamic-static', 'obj info': []},
                      2: {'class': 'press-tap', 'obj info': []},
                      3: {'class': 'dangerous-safe', 'obj info': []}}
        num_correct_class = torch.zeros((1, 4)).to(self.device)
        num_test_class = torch.zeros((1, 4)).to(self.device)
        num_correct = 0.0
        num_tests = 0.0
        test_loss = 0.0
        y_pred = []
        y_test = []
        for batched_graph, label, info in test_dataloader:
            batched_graph = [graph.to(self.device) for graph in batched_graph]
            label = label.to(self.device)
            pred = model(batched_graph)  # forward computation on the batched graph
            # Accuracy and Loss
            J = self.loss(pred, label)
            test_loss += J.detach().item()
            pred = pred > 0.5
            num_correct += (pred == label).sum().item()
            num_tests += (label.shape[0] * label.shape[1])

            # Confusion matrix
            y_pred.append(class_to_label(pred > 0.5))
            y_test.append(class_to_label(label))

            # Per class accuracy
            num_correct_class += (pred == label)
            uncorrect_class = np.where((pred == label).to('cpu').numpy() == False)[1]
            for missclassified in uncorrect_class:
                statistics[missclassified]['obj info'].append(info)
            num_test_class += 1

        acc = num_correct / num_tests
        print(f'Overall accuracy: {acc}, Loss: {test_loss / len(test_dataloader)}')

        num_correct_class = num_correct_class / num_test_class
        classes = ['Big/Small', 'Dynamic/Static', 'Press/Tap', 'Dangerous/Safe']
        for i in range(4):
            print(f'{classes[i]} -> Accuracy: {num_correct_class[0, i]}')

        matrix_confusion(y_test, y_pred, model.name)

        missclassified_obj(statistics, info_encoder, model.name)

    def save(self, file='GNN_frames.tar'):
        torch.save(self.state_dict(), file)

    def load(file):
        model = GConvNetFrames(device)
        model.load_state_dict(torch.load(file))
        print('### Model loaded ###')
        return model


class GConvNetBigGraph(nn.Module):
    def __init__(self,
                 window_size=30,
                 stride_frac=1):
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
        self.hidden2 = nn.Linear(in_features=hidden2, out_features=100, bias=True)
        self.acthidden2 = nn.SiLU()
        self.output = nn.Linear(in_features=100, out_features=output, bias=True)
        self.actout = nn.Sigmoid()

        self.count_parameters()

    def forward(self, graphs, features):
        # define a NN structure using GDL and Torch layers
        x = self.conv1(graphs, features)
        x = self.conv2(graphs, x)
        # x = dgl.nn.SetTransformerEncoder(30, 4, 4, 20, dropouth = 0.9, dropouta=0.9)(graphs, features)
        graphs.ndata['h'] = x
        x = dgl.nn.MaxPooling()(graphs, x)  # dgl.nn.WeightAndSum(500)(graphs, x)#
        x = self.hidden(x)
        x = self.acthidden(x)
        x = nn.Dropout(p=0.2)(x)

        x = self.hidden2(x)
        x = self.acthidden2(x)
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

    def train(self, train_dataloader, validation_dataloader, epochs=70, lr=0.0005, test_rate=0.8):
        model = self  # create a model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # choose an optimizer
        ## Configuring the DataLoader ##
        batch_size = 1
        num_train = train_dataloader.__len__()

        ## Training phase ##
        acc_history = []
        best_acc = 0
        for epoch in range(epochs):
            print('Epoch ', epoch + 1, 'of ', epochs)
            ex = 0
            running_loss = 0.0
            epoch_loss = 0.0
            for batched_graph, label, _ in train_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph,
                             batched_graph.ndata['feature'].float())  # forward computation on the batched graph
                J = self.loss(pred, label)  # calculate the cost function
                optimizer.zero_grad()  # set the gradients to zero
                J.backward()
                optimizer.step()  # backpropagate
                running_loss += J.detach().item()
                ex += batch_size
                if ex % 200 == 0:
                    print(f'{epoch}/{ex + 1} -> Loss: {running_loss / 200}')
                    epoch_loss += running_loss
                    running_loss = 0.0
            # calculate the accuracy on test set and print
            epoch_loss = epoch_loss / num_train
            print(f'Epoch Loss: {epoch_loss}')
            num_correct = 0.0
            num_tests = 0.0
            val_loss = 0.0
            for batched_graph, label, _ in test_dataloader:
                batched_graph = batched_graph[0]
                pred = model(batched_graph,
                             batched_graph.ndata['feature'].float())  # forward computation on the batched graph
                J = self.loss(pred, label)
                val_loss += J.detach().item()
                num_correct += ((pred > 0.5) == label).sum().item()
                num_tests += (label.shape[0] * label.shape[1])
            acc = num_correct / num_tests
            acc_history.append(acc)
            print(f'Test of overall Accuracy: {acc}, Validation Loss: {val_loss}')
            if (acc > best_acc):
                best_acc = acc
                self.save()

        ## Save the accuracy/epochs report ##
        with open('./logfile.txt', 'w') as fp:
            fp.write(json.dumps(acc_history))
            print('### Log saved ###')

        return acc_history

    def evaluation(self, test_dataloader):
        model = self
        print('### Evaluation of the network ###')
        # calculate the accuracy on test set and print
        num_correct_class = torch.zeros((1, 4));
        num_test_class = torch.zeros((1, 4))
        num_correct = 0.0;
        num_tests = 0.0;
        test_loss = 0.0
        y_pred = [];
        y_test = []
        for batched_graph, label, _ in test_dataloader:
            batched_graph = batched_graph[0]
            pred = model(batched_graph,
                         batched_graph.ndata['feature'].float())  # forward computation on the batched graph
            # Accuracy and Loss
            J = self.loss(pred, label)
            test_loss += J.detach().item()
            pred = pred > 0.5
            num_correct += (pred == label).sum().item()
            num_tests += (label.shape[0] * label.shape[1])

            # Confusion matrix
            y_pred.append(class_to_label(pred))
            y_test.append(class_to_label(label))

            # Per Clcass accuracy
            num_correct_class += (pred == label)
            num_test_class += 1

        acc = num_correct / num_tests
        print(f'Overall accuracy: {acc}, Loss: {test_loss}')

        num_correct_class = num_correct_class / num_test_class
        classes = ['Big/Small', 'Dynamic/Static', 'Press/Tap', 'Dangeours/Safe']
        for i in range(4):
            print(f'{classes[i]} -> Accuracy: {num_correct_class[0, i]}')

        matrix_confusion(y_test, y_pred, 'Big Graph')

    def save(self, file='GNN_BG.tar'):
        torch.save(self.state_dict(), file)

    def load(file):
        model = GConvNetBigGraph()
        model.load_state_dict(torch.load(file))
        print('### Model loaded ###')
        return model

    def evaluation_new(self, test_dataloader, info_encoder):
        model = self
        print('### Evaluation of the network ###')
        # calculate the accuracy on test set and print
        statistics = {0: {'class': 'big-small', 'obj info': []},
                      1: {'class': 'dynamic-static', 'obj info': []},
                      2: {'class': 'press-tap', 'obj info': []},
                      3: {'class': 'dangerous-safe', 'obj info': []}}
        num_correct_class = torch.zeros((1, 4))
        num_test_class = torch.zeros((1, 4))
        num_correct = 0.0
        num_tests = 0.0
        test_loss = 0.0
        y_pred = []
        y_test = []
        for batched_graph, label, info in test_dataloader:
            batched_graph = batched_graph[0]
            pred = model(batched_graph,
                         batched_graph.ndata['feature'].float())  # forward computation on the batched graph
            # Accuracy and Loss
            J = self.loss(pred, label)
            test_loss += J.detach().item()
            pred = pred > 0.5
            num_correct += (pred == label).sum().item()
            num_tests += (label.shape[0] * label.shape[1])

            # Confusion matrix
            y_pred.append(pred.int().tolist()[0])
            y_test.append(label.int().tolist()[0])

            # Per class accuracy
            num_correct_class += (pred == label)
            uncorrect_class = np.where((pred == label).to('cpu').numpy() == False)[1]
            for missclassified in uncorrect_class:
                statistics[missclassified]['obj info'].append(info)
            num_test_class += 1

        acc = num_correct / num_tests
        print(f'Overall accuracy: {acc}, Loss: {test_loss / len(test_dataloader)}')

        num_correct_class = num_correct_class / num_test_class
        classes = ['Big/Small', 'Dynamic/Static', 'Press/Tap', 'Dangerous/Safe']
        for i in range(4):
            print(f'{classes[i]} -> Accuracy: {num_correct_class[0, i]}')

        matrix_confusion(y_test, y_pred, 'BigGraph')

        missclassified_obj(statistics, info_encoder, 'BigGraph')

if __name__ == '__main__':
    NET = 'GConvNetBigGraph'
    if NET == 'GConvNetFrames':
        model = GConvNetFrames(device)

        train_dataloader, \
        validation_dataloader, \
        test_dataloader, \
        info_encoder = get_dataloaders_from_csv(window_size=model.window_size, stride_frac=model.stride_frac)

        # acc_hist = model.train_loop(train_dataloader, validation_dataloader)
        # plt.plot(acc_hist)
        # plt.title('accuracy history')
        # plt.savefig(f'images/accuracy_{model.name}.jpg')
        # plt.show()
        model_best = GConvNetFrames.load('./GNN_frames.tar')
        model_best.evaluation(test_dataloader, info_encoder)
    elif NET == 'GConvNetBigGraph':
        model = GConvNetBigGraph()

        train_dataloader, \
        validation_dataloader, \
        test_dataloader, \
        info_encoder = get_dataloaders_from_csv(window_size=model.window_size, stride_frac=model.stride_frac)

        #model.count_parameters()
        #acc_hist = model.train(train_dataloader, validation_dataloader, epochs=70)
        #plt.plot(acc_hist)
        #plt.show()
        model_best = GConvNetBigGraph.load('./GNN_BG.tar')
        model_best.evaluation_new(test_dataloader, info_encoder)