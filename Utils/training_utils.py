import torch

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

def compute_mean_std(dataset):
    # Concatenate all tensors
    all_data = torch.cat([X for X, _ in dataset], dim=0)  # total_frames, feature_dim
    mean = all_data.mean().item()
    std = all_data.std().item()
    return mean, std

def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * factor, min_lr)
        param_group['lr'] = new_lr
        print(f"adjusting learning rate from {old_lr:.6f} to {new_lr:.6f}\n")