import torch, math
from torch.utils.data import DataLoader

def accuracy_fn(y_real, y_pred, padding_index):
    # Flatten inputs 
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)
    
    # Only consider non-padding entries
    mask = (y_real != padding_index)
    correct = torch.eq(y_real[mask], y_pred[mask]).sum().item()
    total = mask.sum().item()
    
    acc = (correct / total) * 100 if total > 0 else 0.0
    return acc

def compute_mean_std(dataloader: DataLoader):
    sum = 0.0
    sum_sq = 0.0
    total_elements = 0

    for X, _ in dataloader:
        # Flatten batch
        X = X.view(-1) 
        
        sum += torch.sum(X).item()
        sum_sq += torch.sum(X**2).item()
        total_elements += X.numel()

    mean = sum / total_elements # mean
    var = (sum_sq / total_elements) - (math.pow(mean, 2)) # variance
    std = math.sqrt(var) # standard deviation

    return mean, std
