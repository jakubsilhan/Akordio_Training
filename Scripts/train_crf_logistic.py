import os, torch, joblib
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from Akordio_Core.net_config import Config, load_config
from Akordio_Core.song_dataset import SongDataset, make_collate_fn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Akordio_Core.chords import Chords, Complexity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from TorchCRF import CRF

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

def train(config: Config):
    """
    Training for a CRF model

    Uses the same configuration as the used recognition model.
    """

    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name, str(config.train.test_fold))

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

            X_tensor = torch.tensor(X, dtype=torch.float32)
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

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        test_tensors.append((X_tensor, y_tensor))

    # Dataset
    train_dataset = SongDataset(train_tensors, config)
    test_dataset = SongDataset(test_tensors, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.model.batch_size,shuffle=True, collate_fn=make_collate_fn(config.train.model.padding_index))
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.model.batch_size, shuffle=False, collate_fn=make_collate_fn(config.train.model.padding_index))

    
    # Load pre-model
    model_path = os.path.join(model_folder, "model.joblib")
    loaded = joblib.load(model_path)
    pre_model: LogisticRegression = loaded["model"]
    scaler: StandardScaler = loaded["scaler"]

    # Initialize CRF
    crf = CRF(num_labels=config.train.model.output).to(device)
    optimizer = optim.Adam(crf.parameters(), lr=config.train.model.learning_rate) # TODO look into a separate learning rate for CRF

    # Training loop
    torch.manual_seed(config.base.random_seed)

    ## Initialization
    train_loss_list = []
    train_accuracy_list = []

    test_loss_list = []
    test_accuracy_list = []

    # Early stopping params
    patience = config.train.model.loss_patience
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model = None
    best_epoch = 0

    pbar = tqdm(range(30), desc="Training Progress") # TODO add proper epoch crf count

    ## Pipeline
    for epoch in pbar:

        ### Initializations
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        ### Train
        for X_batch, y_batch in train_dataloader:
            ####Move to device (GPU or CPU)
            X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
            y_batch = y_batch.to(device)
            mask = (y_batch != config.train.model.padding_index).to(device)

            #### Normalization
            ##### Per Dataset
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            X_scaled = scaler.transform(X_flat)
            X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
            targets = y_batch

            #### 1. Forward pass
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            probs = pre_model.predict_proba(X_flat)
            logits = torch.from_numpy(probs).float()
            logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(device)
            preds = crf.viterbi_decode(logits, mask)  # type: ignore

            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=config.train.model.padding_index)

            #### 2. Loss and accuracy
            log_likelihood = crf(logits, y_batch, mask)
            loss = -log_likelihood.mean()

            acc = accuracy_fn(targets, preds_padded, config.train.model.padding_index)

            train_losses.append(loss.cpu().item())
            train_accuracies.append(acc)

            #### 3. Optimizer zero grad
            optimizer.zero_grad()

            #### 4. Loss backward
            loss.backward()

            #### 5. Optimizer step
            optimizer.step()


        ### Test
        for X_batch, y_batch in test_dataloader:
            #### Move to device (GPU or CPU)
            X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
            y_batch = y_batch.to(device)
            mask = (y_batch != config.train.model.padding_index).to(device)

            #### Normalization
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            X_scaled = scaler.transform(X_flat)
            X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
            targets = y_batch

            #### 1. Forward pass
            with torch.inference_mode():
                batch_size, seq_len, feat_dim = X_batch.shape
                X_flat = X_batch.reshape(-1, feat_dim)
                probs = pre_model.predict_proba(X_flat)                
                logits = torch.from_numpy(probs).float()
                logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(device)

                #### 2. Loss and accuracy
                log_likelihood = crf(logits, y_batch, mask)
                loss = -log_likelihood.mean()

                preds = crf.viterbi_decode(logits, mask)  # type: ignore

            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=config.train.model.padding_index)

            acc = accuracy_fn(targets, preds_padded, config.train.model.padding_index)

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
            checkpoint_path = os.path.join(model_folder, f"crf_checkpoint_epoch_{epoch+1}.pt")
            checkpoint_dict={'model': crf.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses}
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

         # Early stopping
        if test_loss_avg < best_test_loss:
            # Store best values
            best_test_loss = test_loss_avg
            best_model = crf.state_dict()
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

    # Save best model
    model_path = os.path.join(model_folder, "best_crf_model.pt")
    state_dict={'model': best_model, 'optimizer': best_optimizer, 'epoch': best_epoch, 'loss': best_losses}
    torch.save(state_dict, model_path)

    # Save final model
    losses = {
        'train_losses': train_loss_list,
        'train_accuracies': train_accuracy_list,
        'test_losses': test_loss_list,
        'test_accuracies': test_accuracy_list
    }

    model_path = os.path.join(model_folder, "final_crf_model.pt")
    state_dict={'model': crf.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': losses}
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
    figure_path = os.path.join(model_folder, "crf_learning_curve.png")
    plt.savefig(figure_path)
    plt.show()

if __name__=="__main__":
    config = load_config("config.yaml")
    train(config)