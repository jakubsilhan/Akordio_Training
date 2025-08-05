import torch, os
import numpy as np
from Core.config import Config
from Model.Model import Model

def classify_chords(x_tensor: torch.Tensor, config: Config):
    # TODO figure out device
    # device agnostic code
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare data
    # x_tensor.to(device)
    x_tensor = x_tensor.unsqueeze(1)
    mean, std = compute_mean_std(x_tensor)
    x_tensor = (x_tensor-mean)/std

    # Load model
    # model = load_model(config, device)
    model = load_model(config)

    # Inference
    with torch.inference_mode():
        # logits = model(x_tensor, device)
        logits = model(x_tensor)
        preds = torch.softmax(logits, dim=2).argmax(dim=2)

    return preds

# def load_model(config: Config, device: str) -> Model:
#     model = Model(config=config).to(device)
#     model_path = os.path.join("Model", "final_model.pt")
#     loaded = torch.load(model_path, map_location=device)
#     model.load_state_dict(loaded["model"])

#     return model

def load_model(config: Config) -> Model:
    model = Model(config=config)
    model_path = os.path.join("Model", "final_model.pt")
    loaded = torch.load(model_path)
    model.load_state_dict(loaded["model"])

    return model

def compute_mean_std(tensor: torch.Tensor):
    mean = torch.mean(tensor).item()
    square_mean = torch.mean(tensor.pow(2)).item()
    std = np.sqrt(square_mean - mean * mean)

    return mean, std