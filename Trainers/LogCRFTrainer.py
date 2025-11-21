import os, joblib, torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from TorchCRF import CRF
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from Utils.training_utils import accuracy_fn, adjusting_learning_rate
from Trainers.BaseTrainer import BaseTrainer

class LogCRFTrainer(BaseTrainer):
    """Trainer for CRF model using Logistic Regression as pre-model"""
    
    def train(self):
        """Main training loop"""
        # Setup
        torch.manual_seed(self.config.base.random_seed)
        
        # Load data
        train_tensors, test_tensors = self.loader.load_data()
        train_dataloader, test_dataloader = self.create_dataloaders(train_tensors, test_tensors)
        
        # Load pre-trained logistic regression model and scaler
        model_path = os.path.join(self.model_folder, "model.joblib")
        loaded = joblib.load(model_path)
        pre_model: LogisticRegression = loaded["model"]
        scaler: StandardScaler = loaded["scaler"]
        
        # Initialize CRF
        crf = CRF(num_labels=self.config.train.model.output).to(self.device)
        optimizer = optim.Adam(crf.parameters(), lr=self.config.train.model.learning_rate)
        
        # Load checkpoint if exists
        state, _, _ = self.load_checkpoint_if_exists(crf, optimizer, 0.0, 1.0, "crf_")
        
        # Training parameters
        patience = self.config.train.model.loss_patience
        total_epochs = state.epoch + self.config.train.model.epoch_count
        
        try:
            pbar = tqdm(range(state.epoch, total_epochs), desc="Training Progress")
            
            for epoch in pbar:
                state.epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    crf, pre_model, scaler, train_dataloader, optimizer
                )
                
                # Evaluate
                test_loss, test_acc = self.evaluate_epoch(
                    crf, pre_model, scaler, test_dataloader
                )
                
                # Update state
                state.train_loss_list.append(train_loss)
                state.train_accuracy_list.append(train_acc)
                state.test_loss_list.append(test_loss)
                state.test_accuracy_list.append(test_acc)
                
                # Log progress
                tqdm.write(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%\n")
                
                # Checkpointing
                if (epoch + 1) % self.config.train.checkpoint_interval == 0:
                    self.save_checkpoint(state, crf, optimizer, 0.0, 1.0, "crf_")
                
                # Early stopping check
                if test_acc > state.best_test_acc:
                    state.best_test_acc = test_acc
                    state.best_model = crf.state_dict()
                    state.best_optimizer = optimizer.state_dict()
                    state.best_epoch = epoch
                    state.best_losses = {
                        'train_losses': state.train_loss_list.copy(),
                        'train_accuracies': state.train_accuracy_list.copy(),
                        'test_losses': state.test_loss_list.copy(),
                        'test_accuracies': state.test_accuracy_list.copy()
                    }
                    state.epochs_no_improve = 0
                    print(f"New best model with acc: {state.best_test_acc:.2f}% at epoch: {state.best_epoch}\n")
                else:
                    state.epochs_no_improve += 1
                
                if state.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}, test accuracy has not improved for {patience} epochs.\n")
                    break
                
                # Adjust learning rate
                if state.before_acc > test_acc:
                    adjusting_learning_rate(optimizer, factor=0.95, min_lr=5e-6)
                state.before_acc = test_acc
                
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        finally:
            self.save_final_models(state, crf, optimizer, 0.0, 1.0, "crf_")
            self.plot_learning_curves(state)
    
    def train_epoch(self, crf: CRF, pre_model: LogisticRegression, scaler: StandardScaler, 
                   dataloader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        crf.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in dataloader:
            # Convert to numpy for sklearn
            X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
            y_batch = y_batch.to(self.device)
            mask = (y_batch != self.config.train.model.padding_index).to(self.device)
            
            # Normalization with scaler
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            X_scaled = scaler.transform(X_flat)
            X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
            
            # Get logits from logistic regression
            X_flat = X_batch.reshape(-1, feat_dim)
            probs = pre_model.predict_proba(X_flat)
            logits = torch.from_numpy(probs).float()
            logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(self.device)
            
            # Get predictions
            preds = crf.viterbi_decode(logits, mask) # type: ignore
            
            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, 
                                       padding_value=self.config.train.model.padding_index)
            
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
        
        average_loss = sum(losses) / len(losses)
        average_acc = sum(accuracies) / len(accuracies)
        
        return average_loss, average_acc
    
    def evaluate_epoch(self, crf: CRF, pre_model: LogisticRegression, scaler: StandardScaler,
                      dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        crf.eval()
        losses = []
        accuracies = []
        
        with torch.inference_mode():
            for X_batch, y_batch in dataloader:
                # Convert to numpy for sklearn
                X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
                y_batch = y_batch.to(self.device)
                mask = (y_batch != self.config.train.model.padding_index).to(self.device)
                
                # Normalization with scaler
                batch_size, seq_len, feat_dim = X_batch.shape
                X_flat = X_batch.reshape(-1, feat_dim)
                X_scaled = scaler.transform(X_flat)
                X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
                
                # Get logits from logistic regression
                X_flat = X_batch.reshape(-1, feat_dim)
                probs = pre_model.predict_proba(X_flat)
                logits = torch.from_numpy(probs).float()
                logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(self.device)
                
                # Loss
                log_likelihood = crf(logits, y_batch, mask)
                loss = -log_likelihood.mean()
                
                # Get predictions
                preds = crf.viterbi_decode(logits, mask) # type: ignore
                
                # Convert to tensor and pad
                preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
                preds_padded = pad_sequence(preds_tensor, batch_first=True,
                                           padding_value=self.config.train.model.padding_index)
                
                acc = accuracy_fn(y_batch, preds_padded, self.config.train.model.padding_index)
                
                losses.append(loss.cpu().item())
                accuracies.append(acc)
        
        average_loss = sum(losses) / len(losses)
        average_acc = sum(accuracies) / len(accuracies)
        
        return average_loss, average_acc