from pathlib import Path

import requests
from matplotlib import pyplot as plt

class Plotter():
    def __init__(self):
        return

    def plot_scatter(self, X, y):
        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.savefig('scatter.png', format="png")
        plt.show()

    def plot_decision_boundary(self, model, x_train, y_train, x_test, y_test):
        if Path("helper_functions.py").is_file():
            print("helper_functions.py already exists, skipping download")
        else:
            print("Downloading helper_functions.py")
            request = requests.get(
                "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
            with open("helper_functions.py", "wb") as file:
                file.write(request.content)
        from helper_functions import plot_predictions, plot_decision_boundary
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, x_train, y_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, x_test, y_test)
        plt.savefig('decision_boundary.png', format="png")
        plt.show()


    def plot_loss(self, epoch_counts, train_losses, test_losses):
        plt.figure(figsize=(12, 6))
        plt.title("Test and Train Loss")
        plt.plot(epoch_counts, train_losses, label="Training Loss")
        plt.plot(epoch_counts, test_losses, label="Test Loss")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig("test and train loss.png", format="png")
        plt.show()
