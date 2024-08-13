import torch
import trainer
from Generator.generator import Generator
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    generator = Generator(device)
    X_train, X_test, y_train, y_test = generator.generate_multi_class_data()
    trainer.train(X_train, X_test, y_train, y_test, 2, 4, device)


if __name__ == '__main__':
    main()
