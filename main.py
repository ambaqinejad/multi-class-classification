import torch
import trainer
from Generator.generator import Generator
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(device)
    X_train, X_test, y_train, y_test = generator.generate_multi_class_data()
    trainer.train(X_train, X_test, y_train, y_test, num_features=X_train.shape[1], num_classes=len(set(y_train.cpu().numpy())), device=device)


if __name__ == '__main__':
    main()
