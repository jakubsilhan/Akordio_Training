import os, torch, shutil, time, copy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Tuple, List, Dict

from Utils.training_utils import accuracy_fn, compute_mean_std
from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Classes.SongDataset import SongDataset, make_collate_fn
from Neural_Nets.CNN import Model as CNN
from Neural_Nets.CR1 import Model as CR1
from Neural_Nets.CR2 import Model as CR2
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
    total_time: float

class BaseTrainer:
    """
    Training class for basic PyTorch chord recognition models
    """
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = DatasetLoaderService(config)
        self.model_folder = os.path.join(
            Path(__file__).resolve().parent.parent,
            "Models",
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
                    },
                    total_time=0.0
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
                best_losses=prev_losses,
                total_time = loaded.get('total_time', 0.0)
            )
            
            # Rename checkpoint
            checkpoint_name = f"checkpoint_epoch_{start_epoch-1}.pt"
            checkpoint_path = os.path.join(self.model_folder, checkpoint_name)
            shutil.move(best_model_path, checkpoint_path)
            
            return state, train_mean, train_std  

    # Trainer specific setup
    def setup(self):
        """ Base Setup method  
        Must contain definitions for these self.params:
            - self.prefix
            - self.train_loader, self.val_loader
            - self.train_mean, self.train_std
            - self.model
            - self.optimizer
            - self.state
        """
        torch.manual_seed(self.config.base.random_seed)
        # Pathing
        self.prefix = ""

        # Data
        train_tensors, valid_tensors = self.loader.load_data()
        self.train_loader, self.val_loader = self.create_dataloaders(train_tensors, valid_tensors)
        self.train_mean, self.train_std = compute_mean_std(self.train_loader)

        # Model
        self.model = self.create_model()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.train.model.padding_index)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.model.learning_rate, weight_decay=self.config.train.model.weight_decay)
        self.state, self.train_mean, self.train_std = self.load_checkpoint_if_exists(self.model, self.optimizer, self.train_mean, self.train_std)


    # Training
    def train(self):
        """Main training loop"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        self.setup()
        shutil.copy2("config.yaml", self.model_folder)
        torch.manual_seed(self.config.base.random_seed)
                
        # Training parameters
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            threshold=self.config.train.model.loss_delta,
            min_lr=5e-6
        )
        total_epochs = self.state.epoch + self.config.train.model.epoch_count
        
        start_time = time.time()

        try:
            pbar = tqdm(range(self.state.epoch, total_epochs), desc="Training Progress")
            
            for epoch in pbar:
                self.state.epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch()
                
                # Evaluate
                valid_loss, valid_acc = self.evaluate_epoch()

                # Scheduler
                scheduler.step(valid_loss)
                
                # Update state
                self.state.train_loss_list.append(train_loss)
                self.state.train_accuracy_list.append(train_acc)
                self.state.valid_loss_list.append(valid_loss)
                self.state.valid_accuracy_list.append(valid_acc)
                
                # Log progress
                tqdm.write(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}% | valid Loss: {valid_loss:.5f}, valid Acc: {valid_acc:.2f}%\n")
                
                # Checkpointing
                if (epoch + 1) % self.config.train.checkpoint_interval == 0:
                    checkpoint_time = self.state.total_time + (time.time() - start_time)
                    self.save_checkpoint(checkpoint_time)
                
                # Best model evaluation
                if valid_loss < (self.state.best_valid_loss - self.config.train.model.loss_delta):
                    self.state.best_valid_loss = valid_loss
                    self.state.best_model = copy.deepcopy(self.model.state_dict())
                    self.state.best_optimizer = copy.deepcopy(self.optimizer.state_dict())
                    self.state.best_epoch = epoch
                    self.state.best_losses = {
                        'train_losses': self.state.train_loss_list.copy(),
                        'train_accuracies': self.state.train_accuracy_list.copy(),
                        'valid_losses': self.state.valid_loss_list.copy(),
                        'valid_accuracies': self.state.valid_accuracy_list.copy()
                    }
                    self.state.epochs_no_improve = 0
                    print(f"New best model with loss: {self.state.best_valid_loss:.2f} at epoch: {self.state.best_epoch}\n")
                else:
                    self.state.epochs_no_improve += 1
                
                # Early stopping check 
                if self.state.epochs_no_improve >= self.config.train.model.loss_patience:
                    print(f"Early stopping at epoch {epoch+1}, valid accuracy has not improved for {self.config.train.model.loss_patience} epochs.\n")
                    break
                
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        finally:
            total_time = self.state.total_time + (time.time() - start_time)
            self.save_final_models(total_time)
            self.plot_learning_curves()

    def train_final(self, epoch_count: int):
        """Main training loop without validation"""
        # Setup
        self.setup()
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)
        torch.manual_seed(self.config.base.random_seed)
                
        
        start_time = time.time()
        try:
            pbar = tqdm(range(self.state.epoch, epoch_count), desc="Training Progress")
            
            for epoch in pbar:
                self.state.epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch()
                
                # Update state
                self.state.train_loss_list.append(train_loss)
                self.state.train_accuracy_list.append(train_acc)
                
                # Log progress
                tqdm.write(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}%\n")
                
                # Checkpointing
                if (epoch + 1) % self.config.train.checkpoint_interval == 0:
                    checkpoint_time = self.state.total_time + (time.time() - start_time)
                    self.save_checkpoint(checkpoint_time)
                
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        finally:
            total_time = self.state.total_time + (time.time() - start_time)
            self.save_final_models(total_time, final=True)
            self.plot_learning_curves()

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        self.model.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Normalization
            X_batch = (X_batch - self.train_mean) / self.train_std
            
            # Forward pass
            logits = self.model(X_batch)
            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            
            # Loss and accuracy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = y_batch.view(-1)
            
            loss = self.loss_fn(flat_logits, flat_targets)
            acc = accuracy_fn(y_batch, preds, self.config.train.model.padding_index)
            
            losses.append(loss.cpu().item())
            accuracies.append(acc)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 
        
        return average_loss, average_acc
    
    def evaluate_epoch(self) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        self.model.eval()
        losses = []
        accuracies = []
        
        with torch.inference_mode():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Normalization
                X_batch = (X_batch - self.train_mean) / self.train_std
                
                # Forward pass
                logits = self.model(X_batch)
                preds = torch.softmax(logits, dim=2).argmax(dim=2)
                
                # Loss and accuracy
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = y_batch.view(-1)
                
                loss = self.loss_fn(flat_logits, flat_targets)
                acc = accuracy_fn(y_batch, preds, self.config.train.model.padding_index)
                
                losses.append(loss.cpu().item())
                accuracies.append(acc)
        
        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 

        return average_loss, average_acc
    
    # Saving
    def save_checkpoint(self, time: float):
        """Save checkpoint at regular intervals"""
        losses = {
            'train_losses': self.state.train_loss_list,
            'train_accuracies': self.state.train_accuracy_list,
            'valid_losses': self.state.valid_loss_list,
            'valid_accuracies': self.state.valid_accuracy_list
        }
        normalization = {'mean': self.train_mean, 'std': self.train_std}
        
        checkpoint_path = os.path.join(self.model_folder, f"{self.prefix}checkpoint_epoch_{self.state.epoch+1}.pt")
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.state.epoch,
            'loss': losses,
            'normalization': normalization,
            'total_time': time
        }
        torch.save(checkpoint_dict, checkpoint_path)

    def save_final_models(self, time: float, final: bool = False):
        """Save best and final models"""
        normalization = {'mean': self.train_mean, 'std': self.train_std}
        
        # Save best model
        if not final:
            best_model_path = os.path.join(self.model_folder, f"{self.prefix}best_model.pt")
            best_state_dict = {
                'model': self.state.best_model,
                'optimizer': self.state.best_optimizer,
                'epoch': self.state.best_epoch,
                'loss': self.state.best_losses,
                'normalization': normalization,
                'total_time': time
            }
            torch.save(best_state_dict, best_model_path)
            print(f"\nSaving best model from epoch {self.state.best_epoch} with loss {self.state.best_valid_loss:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(self.model_folder, f"{self.prefix}final_model.pt")
        losses = {
            'train_losses': self.state.train_loss_list,
            'train_accuracies': self.state.train_accuracy_list,
            'valid_losses': self.state.valid_loss_list,
            'valid_accuracies': self.state.valid_accuracy_list
        }
        final_state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.state.epoch,
            'loss': losses,
            'normalization': normalization,
            'total_time': time
        }
        torch.save(final_state_dict, final_model_path)

    def plot_learning_curves(self, final: bool = False):
        """Plot and save loss and accuracy curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.state.train_loss_list, label="Train Loss")
        if not final:
            plt.plot(self.state.valid_loss_list, label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.state.train_accuracy_list, label="Train Accuracy")
        if not final:
            plt.plot(self.state.valid_accuracy_list, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.legend()
        
        figure_path = os.path.join(self.model_folder, "learning_curve.png")
        plt.savefig(figure_path)
        plt.show()