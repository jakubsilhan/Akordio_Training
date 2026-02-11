import os, torch, shutil, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Tuple, List, Dict

from Utils.training_utils import accuracy_fn, adjusting_learning_rate, compute_mean_std
from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Classes.SongDataset import SongDataset, make_collate_fn
from Neural_Nets.CNN import Model as CNN
from Neural_Nets.CR1 import Model as CR1
from Neural_Nets.CR2 import Model as CR2
from Neural_Nets.SimpleLSTM import Model as SimpleLSTM
from Neural_Nets.BTC import Model as BTC
from Services.DatasetLoaderService import DatasetLoaderService

@dataclass
class TrainingState:
    """Holds the current state of training"""
    epoch: int
    best_epoch: int
    best_valid_loss: float
    epochs_no_improve: int
    train_loss_list: List[float]
    train_accuracy_list: List[float]
    valid_loss_list: List[float]
    valid_accuracy_list: List[float]
    best_model: Dict
    best_optimizer: Dict
    best_losses: Dict

class BaseTrainer:
    """
    Training class for basic PyTorch chord recognition models
    """
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = DatasetLoaderService(config)
        self.loss_delta = 1e-3
        self.model_folder = os.path.join(
            config.train.model_path, 
            config.train.model_name, 
            str(config.train.val_fold)
        )
    
    def create_dataloaders(self, train_tensors, valid_tensors) -> Tuple[DataLoader, DataLoader]:
        """Create train and valid dataloaders"""
        train_dataset = SongDataset(train_tensors, self.config)
        valid_dataset = SongDataset(valid_tensors, self.config)
        
        collate_fn = make_collate_fn(self.config.train.model.padding_index)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.train.model.batch_size,
            shuffle=True, 
            collate_fn=collate_fn
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=self.config.train.model.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        return train_dataloader, valid_dataloader
    
    # Initializations
    def create_model(self) -> nn.Module:
        """Create and return the appropriate model based on config"""
        model_classes = {
            "BTC": BTC,
            "CR2": CR2,
            "CNN": CNN,
        }
        
        model_type = self.config.train.model_type
        model_class = model_classes.get(model_type, CR1) # Default to CR1
        
        # Copy model file to model folder
        model_file = f"Neural_Nets/{model_type}.py" if model_type in model_classes else "Neural_Nets/CR1.py"
        shutil.copy2(model_file, os.path.join(self.model_folder, "Model.py"))
        
        # Initialize model
        model = model_class(config=self.config, device=self.device).to(self.device)
        return model

    def load_checkpoint_if_exists(self, model: nn.Module, optimizer: optim.Optimizer, train_mean: float, train_std: float, prefix = "") -> Tuple[TrainingState, float, float]:
            """Load checkpoint if it exists, otherwise return fresh training state"""
            best_model_path = os.path.join(self.model_folder, f"{prefix}best_model.pt")
            # Initialize fresh training state if no checkpoint
            if not os.path.exists(best_model_path):
                print("Failed to find best saved model!")
                state = TrainingState(
                    epoch=0,
                    best_epoch=0,
                    best_valid_loss=float('inf'),
                    epochs_no_improve=0,
                    train_loss_list=[],
                    train_accuracy_list=[],
                    valid_loss_list=[],
                    valid_accuracy_list=[],
                    best_model=model.state_dict(),
                    best_optimizer=optimizer.state_dict(),
                    best_losses={
                        'train_losses': [],
                        'train_accuracies': [],
                        'valid_losses': [],
                        'valid_accuracies': []
                    }
                )
                return state, train_mean, train_std
            
            # Load checkpoint data
            loaded = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(loaded['model'])
            optimizer.load_state_dict(loaded['optimizer'])
            
            start_epoch = loaded.get('epoch', 0) + 1
            prev_losses = loaded.get('loss', {})
            normalization = loaded.get('normalization', {})
            
            train_mean = normalization.get('mean', train_mean)
            train_std = normalization.get('std', train_std)

            valid_losses = prev_losses.get('valid_losses', [])
            current_best = min(valid_losses) if valid_losses else float('inf')
            
            state = TrainingState(
                epoch=start_epoch,
                best_epoch=start_epoch - 1,
                best_valid_loss=current_best,
                epochs_no_improve=0,
                train_loss_list=prev_losses.get('train_losses', []),
                train_accuracy_list=prev_losses.get('train_accuracies', []),
                valid_loss_list=prev_losses.get('valid_losses', []),
                valid_accuracy_list=prev_losses.get('valid_accuracies', []),
                best_model=model.state_dict(),
                best_optimizer=optimizer.state_dict(),
                best_losses=prev_losses
            )
            
            # Rename checkpoint
            checkpoint_name = f"checkpoint_epoch_{start_epoch-1}.pt"
            checkpoint_path = os.path.join(self.model_folder, checkpoint_name)
            shutil.move(best_model_path, checkpoint_path)
            
            return state, train_mean, train_std  

    # Training
    def train(self):
        """Main training loop"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)
        torch.manual_seed(self.config.base.random_seed)
        
        # Load data
        train_tensors, valid_tensors = self.loader.load_data()
        train_dataloader, valid_dataloader = self.create_dataloaders(train_tensors, valid_tensors)
        
        # Compute normalization
        train_mean, train_std = compute_mean_std(train_dataloader)
        
        # Create model, loss, optimizer
        model = self.create_model()
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.train.model.padding_index)
        # optimizer = optim.Adam(model.parameters(), lr=self.config.train.model.learning_rate, weight_decay=self.config.train.model.weight_decay, betas=(0.9, 0.98), eps=1e-9)
        optimizer = optim.Adam(model.parameters(), lr=self.config.train.model.learning_rate, weight_decay=self.config.train.model.weight_decay)
        
        # Load checkpoint if exists
        state, train_mean, train_std = self.load_checkpoint_if_exists(model, optimizer, train_mean, train_std)
        
        # Training parameters
        patience = self.config.train.model.loss_patience
        total_epochs = state.epoch + self.config.train.model.epoch_count
        
        start_time = time.time()

        try:
            pbar = tqdm(range(state.epoch, total_epochs), desc="Training Progress")
            
            for epoch in pbar:
                state.epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_dataloader, loss_fn, optimizer, train_mean, train_std
                )
                
                # Evaluate
                valid_loss, valid_acc = self.evaluate_epoch(
                    model, valid_dataloader, loss_fn, train_mean, train_std
                )
                
                # Update state
                state.train_loss_list.append(train_loss)
                state.train_accuracy_list.append(train_acc)
                state.valid_loss_list.append(valid_loss)
                state.valid_accuracy_list.append(valid_acc)
                
                # Log progress
                tqdm.write(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}% | valid Loss: {valid_loss:.5f}, valid Acc: {valid_acc:.2f}%\n")
                
                # Checkpointing
                if (epoch + 1) % self.config.train.checkpoint_interval == 0:
                    checkpoint_time = time.time() - start_time
                    self.save_checkpoint(state, model, optimizer, train_mean, train_std, checkpoint_time)
                
                # Best model evaluation
                if valid_loss < (state.best_valid_loss - self.loss_delta):
                    state.best_valid_loss = valid_loss
                    state.best_model = model.state_dict()
                    state.best_optimizer = optimizer.state_dict()
                    state.best_epoch = epoch
                    state.best_losses = {
                        'train_losses': state.train_loss_list.copy(),
                        'train_accuracies': state.train_accuracy_list.copy(),
                        'valid_losses': state.valid_loss_list.copy(),
                        'valid_accuracies': state.valid_accuracy_list.copy()
                    }
                    state.epochs_no_improve = 0
                    print(f"New best model with loss: {state.best_valid_loss:.2f} at epoch: {state.best_epoch}\n")
                else:
                    state.epochs_no_improve += 1
                
                # Adjust learning rate
                if state.epochs_no_improve > 0 and state.epochs_no_improve % 3 == 0:
                    adjusting_learning_rate(optimizer, factor=0.95, min_lr=5e-6)

                # Early stopping check 
                if state.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}, valid accuracy has not improved for {patience} epochs.\n")
                    break
                
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        finally:
            total_time = time.time() - start_time
            self.save_final_models(state, model, optimizer, train_mean, train_std, total_time)
            self.plot_learning_curves(state)

    def train_epoch(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, train_mean: float, train_std: float) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        model.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Normalization
            X_batch = (X_batch - train_mean) / train_std
            
            # Forward pass
            logits = model(X_batch)
            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            
            # Loss and accuracy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = y_batch.view(-1)
            
            loss = loss_fn(flat_logits, flat_targets)
            acc = accuracy_fn(y_batch, preds, self.config.train.model.padding_index)
            
            losses.append(loss.cpu().item())
            accuracies.append(acc)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 
        
        return average_loss, average_acc
    
    def evaluate_epoch(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, train_mean: float, train_std: float) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        model.eval()
        losses = []
        accuracies = []
        
        with torch.inference_mode():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Normalization
                X_batch = (X_batch - train_mean) / train_std
                
                # Forward pass
                logits = model(X_batch)
                preds = torch.softmax(logits, dim=2).argmax(dim=2)
                
                # Loss and accuracy
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = y_batch.view(-1)
                
                loss = loss_fn(flat_logits, flat_targets)
                acc = accuracy_fn(y_batch, preds, self.config.train.model.padding_index)
                
                losses.append(loss.cpu().item())
                accuracies.append(acc)
        
        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 

        return average_loss, average_acc
    
    # Saving
    def save_checkpoint(self, state: TrainingState, model: nn.Module, optimizer: optim.Optimizer, train_mean: float, train_std: float, time: float, prefix: str = ""):
        """Save checkpoint at regular intervals"""
        losses = {
            'train_losses': state.train_loss_list,
            'train_accuracies': state.train_accuracy_list,
            'valid_losses': state.valid_loss_list,
            'valid_accuracies': state.valid_accuracy_list
        }
        normalization = {'mean': train_mean, 'std': train_std}
        
        checkpoint_path = os.path.join(self.model_folder, f"{prefix}checkpoint_epoch_{state.epoch+1}.pt")
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': state.epoch,
            'loss': losses,
            'normalization': normalization,
            'total_time': time
        }
        torch.save(checkpoint_dict, checkpoint_path)

    def save_final_models(self, state: TrainingState, model: nn.Module, optimizer: optim.Optimizer, train_mean: float, train_std: float, time: float, prefix = ""):
        """Save best and final models"""
        normalization = {'mean': train_mean, 'std': train_std}
        
        # Save best model
        best_model_path = os.path.join(self.model_folder, f"{prefix}best_model.pt")
        best_state_dict = {
            'model': state.best_model,
            'optimizer': state.best_optimizer,
            'epoch': state.best_epoch,
            'loss': state.best_losses,
            'normalization': normalization,
            'total_time': time
        }
        torch.save(best_state_dict, best_model_path)
        print(f"\nSaving best model from epoch {state.best_epoch} with loss {state.best_valid_loss:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(self.model_folder, f"{prefix}final_model.pt")
        losses = {
            'train_losses': state.train_loss_list,
            'train_accuracies': state.train_accuracy_list,
            'valid_losses': state.valid_loss_list,
            'valid_accuracies': state.valid_accuracy_list
        }
        final_state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': state.epoch,
            'loss': losses,
            'normalization': normalization,
            'total_time': time
        }
        torch.save(final_state_dict, final_model_path)

    def plot_learning_curves(self, state: TrainingState):
        """Plot and save loss and accuracy curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(state.train_loss_list, label="Train Loss")
        plt.plot(state.valid_loss_list, label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(state.train_accuracy_list, label="Train Accuracy")
        plt.plot(state.valid_accuracy_list, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.legend()
        
        figure_path = os.path.join(self.model_folder, "learning_curve.png")
        plt.savefig(figure_path)
        plt.show()