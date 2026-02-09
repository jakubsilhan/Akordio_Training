import os, shutil, torch, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from TorchCRF import CRF
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple

from Utils.training_utils import accuracy_fn, adjusting_learning_rate
from Akordio_Core.Classes.SongDataset import SongDataset
from Trainers.BaseTrainer import BaseTrainer

class CRFTrainer(BaseTrainer):
    """
    Training class TorchCRF model used with a PyTorch chord recognition model
    """
    def train(self):
        """Main training loop"""
        # Setup
        torch.manual_seed(self.config.base.random_seed)
        
        # Load data
        train_tensors, valid_tensors = self.loader.load_data()
        train_dataloader, valid_dataloader = self.create_dataloaders(train_tensors, valid_tensors)
        
        # Compute normalization
        train_dataset = SongDataset(train_tensors, self.config)
        
        # Create and load model
        pre_model = self.create_model()
        model_path = os.path.join(self.model_folder, "best_model.pt")
        loaded = torch.load(model_path, map_location=self.device)
        pre_model.load_state_dict(loaded['model'])
        normalization = loaded['normalization']
        train_mean = normalization['mean']
        train_std = normalization['std']

        pre_model.eval()
        for p in pre_model.parameters():
            p.requires_grad = False

        # Initialize CRF
        crf = CRF(num_labels=self.config.train.model.output).to(self.device)
        optimizer = optim.Adam(crf.parameters(), lr=self.config.train.model.learning_rate)

        # Load checkpoint if exists
        state, train_mean, train_std = self.load_checkpoint_if_exists(crf, optimizer, train_mean, train_std, "crf_")

        # Training parameters
        patience = self.config.train.model.loss_patience
        total_epochs = state.epoch + self.config.train.model.epoch_count
        
        start_time = time.time()

        try:
            pbar = tqdm(range(state.epoch, total_epochs), desc="Training Progress")
            
            for epoch in pbar:
                state.epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch(crf, pre_model, train_dataloader, optimizer, train_mean, train_std)
                
                # Evaluate
                valid_loss, valid_acc = self.evaluate_epoch(crf, pre_model, valid_dataloader, train_mean, train_std)
                
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
                    self.save_checkpoint(state, crf, optimizer, train_mean, train_std, checkpoint_time, "crf_")
                
                # Early stopping check
                if valid_loss < (state.best_valid_loss - self.loss_delta):
                    state.best_valid_loss = valid_loss
                    state.best_model = crf.state_dict()
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
            self.save_final_models(state, crf, optimizer, train_mean, train_std, total_time, "crf_")
            self.plot_learning_curves(state)

    def train_epoch(self, crf: CRF, pre_model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, train_mean: float, train_std: float) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        crf.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mask = (y_batch != self.config.train.model.padding_index).to(self.device)
            
            # Normalization
            X_batch = (X_batch - train_mean) / train_std
            
            # Forward pass
            with torch.inference_mode():
                logits = pre_model(X_batch)
            preds = crf.viterbi_decode(logits, mask)
            
            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=self.config.train.model.padding_index)

            # Loss and accuracy
            log_likelihood = crf(logits, y_batch, mask)
            loss = -log_likelihood.mean()
            
            acc = accuracy_fn(y_batch, preds_padded, self.config.train.model.padding_index)
            
            losses.append(loss.cpu().item())
            accuracies.append(acc)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 
        
        return average_loss, average_acc
    
    def evaluate_epoch(self, crf: CRF, pre_model: nn.Module, dataloader: DataLoader, train_mean: float, train_std: float) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        crf.eval()
        losses = []
        accuracies = []
        
        with torch.inference_mode():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                mask = (y_batch != self.config.train.model.padding_index).to(self.device)
                
                # Normalization
                X_batch = (X_batch - train_mean) / train_std
                
                # Forward pass
                logits = pre_model(X_batch)

                #Loss and accuracy
                log_likelihood = crf(logits, y_batch, mask)
                loss = -log_likelihood.mean()

                preds = crf.viterbi_decode(logits, mask)
                
                # Convert to tensor and pad
                preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
                preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=self.config.train.model.padding_index)
                
                acc = accuracy_fn(y_batch, preds_padded, self.config.train.model.padding_index)
                
                losses.append(loss.cpu().item())
                accuracies.append(acc)
        
        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 

        return average_loss, average_acc
