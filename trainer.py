import torch
from Models.multi_class_model import MultiClassModel
import metrics
from GraphPlotter.plotter import Plotter

def train(X_train, X_test, y_train, y_test, num_features, num_classes, device):
    # y_train = y_train.long()
    # y_test = y_test.long()
    model = MultiClassModel(input_features=num_features,
                        output_features=num_classes,
                        hidden_units=8).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    epochs = 1000
    epoch_counts = []
    train_losses = []
    test_losses = []
    # print(model(X_train)[:5])
    for epoch in range(epochs):
        model.train()
        train_logits = model(X_train)
        train_pred = torch.softmax(train_logits, dim=1).argmax(dim=1)
        acc = metrics.accuracy(y_train, train_pred)
        train_loss = loss_fn(train_logits, y_train.long())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_acc = metrics.accuracy(y_test, test_pred)
                test_loss = loss_fn(test_logits, y_test)
                epoch_counts.append(epoch)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"Epoch: {epoch} | Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
    plotter = Plotter()
    plotter.plot_loss(epoch_counts, train_losses, test_losses)
    plotter.plot_decision_boundary(model, X_train, y_train, X_test, y_test)
