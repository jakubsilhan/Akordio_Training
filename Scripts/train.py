import os, torch, shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from Akordio_Core.net_config import Config, load_config
from Akordio_Core.song_dataset import SongDataset, make_collate_fn
from torch.utils.data import DataLoader
from Neural_Nets.CNN import Model as CNN
from Neural_Nets.FifthNet import Model as FifthNet
from Neural_Nets.CR1 import Model as CR1
from Neural_Nets.CR2 import Model as CR2
from Neural_Nets.SimpleLSTM import Model as SimpleLSTM
from Neural_Nets.BTC import Model as BTC
from Akordio_Core.chords import Chords, Complexity

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
        print('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))

def train(config: Config):
    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name, str(config.train.test_fold))
    os.makedirs(model_folder, exist_ok=True)
    shutil.copy2("config.yaml", model_folder)

    match config.train.model_complexity:
        case "complex":
            complexity = Complexity.COMPLEX
        case "majmin7":
            complexity = Complexity.MAJMIN7
        case _:
            complexity = Complexity.MAJMIN

    chord_tool = Chords()

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_tensors = []
    test_tensors = []

    ## Check for dataset
    # TODO add the check here

    ## Load Train
    # TODO add checks for missing folds
    for fold in tqdm(os.listdir(config.train.data_source), desc="Loading train folds"):
        if fold == "config.yaml":
            continue
        if fold == str(config.train.test_fold - 1):
            continue

        fold_dir = os.path.join(config.train.data_source, fold)
        for fragment in os.listdir(fold_dir):
            if not fragment.endswith(".npz"):
                continue
            fragment_path = os.path.join(fold_dir, fragment)
            data = np.load(fragment_path)
            X = data["X"]
            y_raw = data["y"]

            y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
                for chord in y_raw]

            X_tensor = torch.tensor(X, dtype=torch.float64)
            y_tensor = torch.tensor(y, dtype=torch.long)
            train_tensors.append((X_tensor, y_tensor))

    ## Load Test
    test_fold_path = os.path.join(config.train.data_source, str(config.train.test_fold - 1))
    for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
        if not fragment.endswith(".npz"):
            continue

        if "_shift00_" not in fragment:
            continue

        fragment_path = os.path.join(test_fold_path, fragment)
        data = np.load(fragment_path)
        X = data["X"]
        y_raw = data["y"]

        y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
            for chord in y_raw]

        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.long)
        test_tensors.append((X_tensor, y_tensor))

    # Dataset
    train_dataset = SongDataset(train_tensors, config)
    test_dataset = SongDataset(test_tensors, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.model.batch_size,shuffle=True, collate_fn=make_collate_fn(config.train.model.padding_index))
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.model.batch_size, shuffle=False, collate_fn=make_collate_fn(config.train.model.padding_index))

    # Model
    match config.train.model_type:
        case "SimpleLSTM":
            shutil.copy2("Neural_Nets/SimpleLSTM.py", model_folder+"/Model.py")
            model = SimpleLSTM(
                config=config,
                device=device
            ).to(device)
        case "BTC":
            shutil.copy2("Neural_Nets/BTC.py", model_folder+"/Model.py")
            model = BTC(
                config=config, 
                device=device
            ).to(device)
        case "CR2":
            shutil.copy2("Neural_Nets/CR2.py", model_folder+"/Model.py")
            model = CR2(
                config=config, 
                device=device
            ).to(device)
        case "CNN":
            shutil.copy2("Neural_Nets/CNN.py", model_folder+"/Model.py")
            model = CNN(
                config=config, 
                device=device
            ).to(device)
        case "FifthNet":
            shutil.copy2("Neural_Nets/FifthNet.py", model_folder+"/Model.py")
            model = FifthNet(
                config=config, 
                device=device
            ).to(device)
        case _:
            shutil.copy2("Neural_Nets/CR1.py", model_folder+"/Model.py")
            model = CR1(
                config=config,
                device=device
            ).to(device)

    # Loss and optimizer
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.train.model.padding_index)
    optimizer = optim.Adam(model.parameters(), lr=config.train.model.learning_rate, weight_decay=config.train.model.weight_decay, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.train.model.scheduler_step, gamma=config.train.model.scheduler_gamma)

    # data info
    train_mean, train_std = compute_mean_std(train_dataset)

    # Training loop
    torch.manual_seed(config.base.random_seed)

    # Check and load pretrained (trains additional epoch_count)
    final_model_path = os.path.join(model_folder, "best_model.pt")
    if os.path.exists(final_model_path):
            # Load model and optimizer
            model_path = os.path.join(model_folder, "best_model.pt")
            loaded: dict = torch.load(model_path, map_location=device)
            model.load_state_dict(loaded['model'])
            optimizer.load_state_dict(loaded['optimizer'])

            # Load epoch and loss details
            start_epoch = loaded.get('epoch', 0) + 1
            prev_losses: dict = loaded.get('loss', {})

            train_loss_list = prev_losses.get('train_losses', [])
            train_accuracy_list = prev_losses.get('train_accuracies', [])
            test_loss_list = prev_losses.get('test_losses', [])
            test_accuracy_list = prev_losses.get('test_accuracies', [])

            # Load normalization
            normalization:dict = loaded.get('normalization', {})
            train_mean = normalization.get('mean', train_mean)
            train_std = normalization.get('std', train_std)

            # Rename loaded dict to checkpoint
            checkpoint_name = f"checkpoint_epoch_{start_epoch-1}.pt"
            checkpoint_path = os.path.join(model_folder, checkpoint_name)
            shutil.move(model_path, checkpoint_path)

    # Initialization
    start_epoch = 0
    epoch = start_epoch
    best_model = None
    best_epoch = 0
    before_acc = 0
    train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = [], [], [], []

    # Early stopping params
    patience = config.train.model.loss_patience
    best_test_acc = float(0)
    epochs_no_improve = 0

    pbar = tqdm(range(start_epoch, start_epoch + config.train.model.epoch_count), desc="Training Progress")
    ## Pipeline
    for epoch in pbar:
        model.train()

        ### Initializations
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        ### Train
        for X_batch, y_batch in train_dataloader:
            ####Move to device (GPU or CPU)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #### Normalization
            ##### Per Dataset
            X_batch = (X_batch-train_mean)/train_std
            targets = y_batch

            #### 1. Forward pass
            logits = model(X_batch)
            preds = torch.softmax(logits, dim=2).argmax(dim=2)

            #### 2. Loss and accuracy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)

            loss = loss_fn(flat_logits, flat_targets)
            acc = accuracy_fn(targets, preds, config.train.model.padding_index)

            train_losses.append(loss.cpu().item())
            train_accuracies.append(acc)

            #### 3. Optimizer zero grad
            optimizer.zero_grad()

            #### 4. Loss backward
            loss.backward()

            #### 5. Optimizer step
            optimizer.step()

        # scheduler.step()

        ### Test
        for X_batch, y_batch in test_dataloader:
            #### Move to device (GPU or CPU)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #### Normalization
            X_batch = (X_batch-train_mean)/train_std
            targets = y_batch

            with torch.inference_mode():
                #### 1. Forward pass
                logits = model(X_batch)
                preds = torch.softmax(logits, dim=2).argmax(dim=2)

                #### 2. Loss and accuracy
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)

                loss = loss_fn(flat_logits, flat_targets)
                acc = accuracy_fn(targets, preds, config.train.model.padding_index)

                test_losses.append(loss.cpu().item())
                test_accuracies.append(acc)

        ### Checkpointing
        if (epoch + 1) % config.train.checkpoint_interval == 0:
            losses = {
                'train_losses': train_loss_list,
                'train_accuracies': train_accuracy_list,
                'test_losses': test_loss_list,
                'test_accuracies': test_accuracy_list
            }
            normalization = {
                'mean': train_mean,
                'std': train_std
            }
            checkpoint_path = os.path.join(model_folder, f"checkpoint_epoch_{epoch+1}.pt")
            checkpoint_dict={'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses, 'normalization': normalization}
            torch.save(checkpoint_dict, checkpoint_path)

        ### Characteristics
        if epoch % 1 == 0:
            train_loss_avg = sum(train_losses)/len(train_losses)
            train_acc_avg = sum(train_accuracies)/len(train_accuracies)
            test_loss_avg = sum(test_losses)/len(test_losses)
            test_acc_avg = sum(test_accuracies)/len(test_accuracies)

            current_acc = test_acc_avg

            train_loss_list.append(train_loss_avg)
            train_accuracy_list.append(train_acc_avg)
            test_loss_list.append(test_loss_avg)
            test_accuracy_list.append(test_acc_avg)
            tqdm.write(f"Epoch: {epoch} | Loss: {train_loss_avg:.5f}, Acc: {train_acc_avg:.2f}% | Test Loss: {test_loss_avg:.5f}, Test Acc: {test_acc_avg:.2f}%\n")
        
        # Early stopping
        if test_acc_avg > best_test_acc:
            # Store best values
            best_test_acc = test_acc_avg
            best_model = model.state_dict()
            best_epoch = epoch
            best_losses = {
                'train_losses': train_loss_list,
                'train_accuracies': train_accuracy_list,
                'test_losses': test_loss_list,
                'test_accuracies': test_accuracy_list
            }
            best_optimizer = optimizer.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}, test loss has not improved for {patience} epochs.")
            break

        if before_acc > current_acc:
            adjusting_learning_rate(optimizer=optimizer, factor=0.95, min_lr=5e-6)
        before_acc = current_acc

    # Save best model
    normalization = {
                'mean': train_mean,
                'std': train_std
    }
    model_path = os.path.join(model_folder, "best_model.pt")
    state_dict={'model': best_model, 'optimizer': best_optimizer, 'epoch': best_epoch, 'loss': best_losses, 'normalization': normalization}
    torch.save(state_dict, model_path)

    # Save final model
    losses = {
        'train_losses': train_loss_list,
        'train_accuracies': train_accuracy_list,
        'test_losses': test_loss_list,
        'test_accuracies': test_accuracy_list
    }
    normalization = {
                'mean': train_mean,
                'std': train_std
    }

    model_path = os.path.join(model_folder, "final_model.pt")
    state_dict={'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses, 'normalization': normalization}
    torch.save(state_dict, model_path)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label="Train Accuracy")
    plt.plot(test_accuracy_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    figure_path = os.path.join(model_folder, "learning_curve.png")
    plt.savefig(figure_path)
    plt.show()

if __name__=="__main__":
    config = load_config("config.yaml")
    train(config)