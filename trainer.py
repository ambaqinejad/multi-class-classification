import torch
from torch import nn
from Models.multi_class_model import MultiClassModel
from GraphPlotter.plotter import Plotter
from pathlib import Path

def train(X_train, X_test, y_train, y_test, num_features, num_classes, device):
    model = MultiClassModel(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    epochs_count = []
    train_losses = []
    test_losses = []
    epochs = 1000
    print(len(y_train))
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        _, predicted = torch.max(logits.data, 1)
        correct_predictions = torch.eq(predicted, y_train).sum().item()
        accuracy = correct_predictions / len(y_train) * 100
        train_loss = criterion(logits, y_train)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test)
                _, test_predicted = torch.max(test_logits.data, dim=1)
                test_correct_predictions = torch.eq(test_predicted, y_test).sum().item()
                test_accuracy = test_correct_predictions / len(y_test) * 100
                test_loss = criterion(test_logits, y_test)
                epochs_count.append(epoch)
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {accuracy}")
                print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
                print("---------------------")
    plotter = Plotter()
    plotter.plot_loss(epochs_count, train_losses, test_losses)
    plotter.plot_decision_boundary(model, X_train, y_train, X_test, y_test)

    model_directory = Path("ModelDirectory")
    model_directory.mkdir(parents=True, exist_ok=True)
    model_name = "multi-class-classification-v0.pth"
    model_path = model_directory / model_name
    torch.save(obj=model.state_dict(), f=model_path)