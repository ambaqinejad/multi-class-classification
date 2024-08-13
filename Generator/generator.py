from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

from GraphPlotter.plotter import Plotter


class Generator():
    def __init__(self, device):
        self.device = device

    def generate_multi_class_data(self):
        NUM_CLASSES = 5
        NUM_FEATURES = 2
        RANDOM_SEED = 42
        X, y = make_blobs(n_samples=10000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=.5, random_state=RANDOM_SEED)
        plotter = Plotter()
        plotter.plot_scatter(X, y)
        dataframe = pd.DataFrame({
            "X1": X[:, 0],
            "X2": X[:, 1],
            "y": y
        })
        X = torch.from_numpy(X).type(torch.float).to(self.device)
        y = torch.from_numpy(y).type(torch.float).to(self.device)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
        return X_train, X_test, y_train, y_test
