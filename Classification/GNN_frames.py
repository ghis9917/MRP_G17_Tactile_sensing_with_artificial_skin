import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def visualize_loss(history):
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
    plt.savefig(f'loss.jpg')
    # plt.show()
    plt.close()


def matrix_confusion(y_test, y_pred, model):
    # print("y_test:")
    # print(y_test)
    # print("y_pred:")
    # print(y_pred)

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
    # plt.show()


def training(train_loader, val_loader, device, windows, features):
    model = GConvNet(windows, features).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        'loss': [],
        'val_loss': []
    }
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            # get inputs
            graphs, target = data
            graph = [graph.to(device) for graph in graphs]
            target = target.to(torch.float32).to(device)

            # zero the gradients
            optimizer.zero_grad()

            # perform forward pass
            output = model(graph)

            # compute loss
            loss = loss_function(output, target)  # torch.round(torch.sigmoid(output))

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # print statistics
            running_loss += loss.detach().item()  # loss.item() returns the average loss for each sample within the batch
            if i % 1504 == 1503:  # print every 2000 mini-batches
                print(f'[{epoch}, {i + 1}] loss: {running_loss / 1054}')
                epoch_loss += running_loss
                running_loss = 0.0

        # epoch_loss = running_loss / len(train_loader)  # TODO running loss is set to zero!
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, loss: {epoch_loss}")
        history['loss'].append(epoch_loss)

        model.eval()

        val_loss, correct, total = 0.0, 0.0, len(val_loader)
        # Iterate over the test data and generate predictions
        for i, data in enumerate(val_loader, 0):
            # Get inputs
            graph, target = data
            graph = [graph.to(device) for graph in graphs]
            target = target.to(torch.float32).to(device)

            # Generate outputs
            output = model(graph)
            # prediction = torch.round(torch.sigmoid(output))

            # Compute loss
            loss = loss_function(output, target)
            val_loss += loss.detach().item()  # * inputs.size(0) TODO Check what to use
            # val_loss += loss.item().len(inputs)

            # update correct
            correct += 1 if torch.round(output) == target else 0

        accuracy = 100.0 * correct / total
        val_loss = val_loss / len(val_loader)
        print(f"Val Loss for epoch {epoch}: {val_loss}, accuracy: {accuracy}")
        history['val_loss'].append(val_loss)

    # Process is complete.
    print('Training process has finished. Saving trained model.')
    torch.save(model, f'model_GraphConv.pth')
    visualize_loss(history)


def evaluate(test_loader, device):
    model = torch.load('model_GraphConv.pth').to(device)
    print('--------Testing-------')
    correct, pred, true, true_pos, total = 0, 0, 0, 0, len(test_loader)
    model.eval()
    y_test, y_pred = [], []
    # Iterate over the test data and generate predictions
    for i, data in enumerate(test_loader, 0):
        # Get inputs
        graph, target = data

        # Generate outputs
        graph = graph.to(device)
        target = target.to(torch.float32).to(device)

        # Generate outputs
        output = model(graph, graph.ndata['feature'])
        prediction = torch.round(output)

        # Set total and correct
        correct += 1 if prediction == target else 0
        true_pos += 1 if prediction == target and target == 1 else 0
        true += 1 if target == 1 else 0
        pred += 1 if prediction == 1 else 0
        y_test.append(target.item())
        y_pred.append(prediction.item())

    print('tot: ', total, ', correct: ', correct, ', predicted 0s: ', pred, ', true 0s: ', true)
    accuracy = 100.0 * correct / total
    precision = true_pos / pred if pred else 0.0  # TP / TP + FP
    recall = true_pos / true if true else 0.0
    fb1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    matrix_confusion(y_test, y_pred, 'SimpleGConv')

    # Print accuracy
    print(f'Evaluation: accuracy {accuracy},' f' precision {precision}, recall {recall}, f-score {fb1}')
    print('--------------------------------')


if __name__ == '__main__':
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: ', device)

    parameters = {
        'window_size': 1,
        'stride_frac': 1
    }

    train_loader, val_loader, test_loader = get_dataloaders_from_csv(**parameters)

    windows = int(30/int(parameters['stride_frac'] * parameters['window_size']))

    training(train_loader, val_loader, device, windows, parameters['window_size'])
    evaluate(test_loader, device)
