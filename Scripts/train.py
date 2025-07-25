import os, torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Core.config import Config, load_config
from Core.song_dataset import SongDataset, make_collate_fn
from torch.utils.data import DataLoader
from Neural_Nets.CRNN import CRNN
from Utils.chords import Chords, Complexity

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

def compute_mean_std(dataloader):
    mean = 0.0
    square_mean = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        mean += torch.mean(X_batch).item()
        square_mean += torch.mean(X_batch.pow(2)).item()
        num_batches += 1

    mean /= num_batches
    square_mean /= num_batches
    std = np.sqrt(square_mean - mean * mean)

    return mean, std

def train(config: Config):
    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name)
    os.makedirs(model_folder, exist_ok=True)

    match config.train.model_complexity:
        case "complex":
            complexity = Complexity.COMPLEX
        case "majmin7":
            complexity = Complexity.MAJMIN7
        case default:
            complexity = Complexity.MAJMIN

    chord_tool = Chords()

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    # df_list_train = []
    # df_list_test = []

    train_tensors = []
    test_tensors = []

    ## Check for dataset
    # TODO add the check here

    ## Load Train
    for fold in tqdm(os.listdir(config.train.data_source), desc="Loading train folds"):
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

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            train_tensors.append((X_tensor, y_tensor))

    ## Load Test
    test_fold_path = os.path.join(config.train.data_source, str(config.train.test_fold - 1))
    for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
        if not fragment.endswith(".npz"):
            continue
        fragment_path = os.path.join(test_fold_path, fragment)
        data = np.load(fragment_path)
        X = data["X"]
        y_raw = data["y"]

        y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
            for chord in y_raw]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        test_tensors.append((X_tensor, y_tensor))

    # ## Load Train
    # for fold in tqdm(os.listdir(config.data.preprocessed_dir), desc="Loading train folds"):
    #     if fold == str(config.train.test_fold-1):
    #         continue
        
    #     fold_dir = os.path.join(config.data.preprocessed_dir, fold)
    #     for fragment in os.listdir(fold_dir):
    #         fragment_path = os.path.join(fold_dir, fragment)
    #         fragment_df = pd.read_csv(fragment_path)
    #         df_list_train.append(fragment_df)

    # ## Load Test
    # test_fold_path = os.path.join(config.data.preprocessed_dir, str(config.train.test_fold-1))
    # for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
    #     fragment_path = os.path.join(test_fold_path, fragment)
    #     fragment_df = pd.read_csv(fragment_path)
    #     df_list_test.append(fragment_df)


    # # Convert to NumPy
    # train_tensors = []
    # test_tensors = []

    # ## Train
    # for fragment_df in tqdm(df_list_train, desc="Converting train dataset"):
    #     X = fragment_df.iloc[:,1:config.train.model.input+1].values
    #     # Convert label to proper number (accounting for chord complexity)
    #     y = fragment_df["chord"].apply(
    #         lambda chord: chord_tool.encode(
    #             chord=chord_tool.reduce(chord, complexity),
    #             type=complexity
    #         )
    #     ).values
    #     X_tensor = torch.tensor(X, dtype=torch.float32)
    #     y_tensor = torch.tensor(y, dtype=torch.long)

    #     train_tensors.append((X_tensor, y_tensor))

    # for fragment_df in tqdm(df_list_test, desc="Converting test dataset"):
    #     X = fragment_df.iloc[:,1:config.train.model.input+1].values
    #     # Convert label to proper number (accounting for chord complexity)
    #     y = fragment_df["chord"].apply(
    #         lambda chord: chord_tool.encode(
    #             chord=chord_tool.reduce(chord, complexity),
    #             type=complexity
    #         )
    #     ).values
    #     X_tensor = torch.tensor(X, dtype=torch.float32)
    #     y_tensor = torch.tensor(y, dtype=torch.long)

    #     test_tensors.append((X_tensor, y_tensor))

    # Dataset
    train_dataset = SongDataset(train_tensors)
    test_dataset = SongDataset(test_tensors)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.model.batch_size,shuffle=True, collate_fn=make_collate_fn(config.train.model.padding_index))
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.model.batch_size, shuffle=False, collate_fn=make_collate_fn(config.train.model.padding_index))

    # Model
    match config.train.model_type:
        case default:
            model = CRNN(
                feature_size=config.train.model.input,
                output_features=config.train.model.output,
                hidden_size=config.train.model.hidden,
                num_layers=config.train.model.layers,
                bidirectional=config.train.model.bidirectional,
                device=device
            )

    # Loss and optimizer
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.train.model.padding_index)
    optimizer = optim.Adam(model.parameters(), lr=config.train.model.learning_rate, weight_decay=config.train.model.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.train.model.scheduler_step, gamma=config.train.model.scheduler_gamma)

    # TODO add checkpoint load here if able


    # data info
    train_mean, train_std = compute_mean_std(train_dataloader)
    test_mean, test_std = compute_mean_std(test_dataloader)

    # Training loop
    torch.manual_seed(config.base.random_seed)

    ## Initialization
    train_loss_list = []
    train_accuracy_list = []

    test_loss_list = []
    test_accuracy_list = []

    pbar = tqdm(range(config.train.model.epoch_count), desc="Training Progress")

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
            X_batch = (X_batch-train_mean)/train_std
            targets = y_batch

            #### 1. Forward pass
            logits = model(X_batch, device)
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

        scheduler.step()

        ### Test
        for X_batch, y_batch in test_dataloader:
            #### Move to device (GPU or CPU)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #### Normalization
            X_batch = (X_batch-test_mean)/test_std
            targets = y_batch

            with torch.inference_mode():
                #### 1. Forward pass
                logits = model(X_batch, device)
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
            checkpoint_path = os.path.join(model_folder, f"checkpoint_epoch_{epoch+1}.pt")
            checkpoint_dict={'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses}
            torch.save(checkpoint_dict, checkpoint_path)

        ### Characteristics
        if epoch % 1 == 0:
            train_loss_avg = sum(train_losses)/len(train_losses)
            train_acc_avg = sum(train_accuracies)/len(train_accuracies)
            test_loss_avg = sum(test_losses)/len(test_losses)
            test_acc_avg = sum(test_accuracies)/len(test_accuracies)

            train_loss_list.append(train_loss_avg)
            train_accuracy_list.append(train_acc_avg)
            test_loss_list.append(test_loss_avg)
            test_accuracy_list.append(test_acc_avg)
            tqdm.write(f"Epoch: {epoch} | Loss: {train_loss_avg:.5f}, Acc: {train_acc_avg:.2f}% | Test Loss: {test_loss_avg:.5f}, Test Acc: {test_acc_avg:.2f}%\n")

    losses = {
        'train_losses': train_loss_list,
        'train_accuracies': train_accuracy_list,
        'test_losses': test_loss_list,
        'test_accuracies': test_accuracy_list
    }

    # Save model
    model_path = os.path.join(model_folder, "final_model.pt")
    state_dict={'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses}
    torch.save(state_dict, model_path)


if __name__=="__main__":
    config = load_config("config.yaml")
    train(config)