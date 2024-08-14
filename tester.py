import torch
from Generator.generator import Generator
from pathlib import Path
from Models.multi_class_model import MultiClassModel


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(device)
    X_train, X_test, y_train, y_test = generator.generate_multi_class_data()
    model_directory = Path("./ModelDirectory")
    model_name = "multi-class-classification-v0.pth"
    model_path = model_directory / model_name
    model = MultiClassModel(X_train.shape[1], len(set(y_train.cpu().numpy()))).to(device)
    model.load_state_dict(torch.load(f=model_path))
    with torch.inference_mode():
        model.eval()
        output = model(X_test)
        _, predicted_label = torch.max(output, 1)
        correct_predictions = torch.eq(predicted_label, y_test).sum().item()
        accuracy = correct_predictions / len(y_test) * 100
        print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test()
