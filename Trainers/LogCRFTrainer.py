import os, joblib, torch, time
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
    
    def setup(self):
        torch.manual_seed(self.config.base.random_seed)
        # Pathing
        self.prefix = "crf_"

        # Data
        train_tensors, valid_tensors = self.loader.load_data()
        self.train_loader, self.val_loader = self.create_dataloaders(train_tensors, valid_tensors)

        # Pre-Model
        pre_model_path = os.path.join(self.model_folder, "model.joblib")
        pre_model_loaded = joblib.load(pre_model_path)
        self.pre_model: LogisticRegression = pre_model_loaded["model"]
        self.scaler: StandardScaler = pre_model_loaded["scaler"]

        # CRF
        self.model = CRF(num_labels=self.config.train.model.output).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.model.learning_rate)
        self.train_mean, self.train_std = 0.0, 1.0
        self.state, self.train_mean, self.train_std = self.load_checkpoint_if_exists(self.model, self.optimizer, self.train_mean, self.train_std, "crf_")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        self.model.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in self.train_loader:
            # Convert to numpy for sklearn
            X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
            y_batch = y_batch.to(self.device)
            mask = (y_batch != self.config.train.model.padding_index).to(self.device)
            
            # Normalization with scaler
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            X_scaled = self.scaler.transform(X_flat)
            X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
            
            # Get logits from logistic regression
            X_flat = X_batch.reshape(-1, feat_dim)
            probs = self.pre_model.predict_proba(X_flat)
            logits = torch.from_numpy(probs).float()
            logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(self.device)
            
            # Get predictions
            preds = self.model.viterbi_decode(logits, mask) # type: ignore
            
            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, 
                                       padding_value=self.config.train.model.padding_index)
            
            # Loss and accuracy
            log_likelihood = self.model(logits, y_batch, mask)
            loss = -log_likelihood.mean()

            acc = accuracy_fn(y_batch, preds_padded, self.config.train.model.padding_index)
            
            losses.append(loss.cpu().item())
            accuracies.append(acc)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        average_loss = sum(losses) / len(losses)
        average_acc = sum(accuracies) / len(accuracies)
        
        return average_loss, average_acc
    
    def evaluate_epoch(self) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        self.model.eval()
        losses = []
        accuracies = []
        
        with torch.inference_mode():
            for X_batch, y_batch in self.train_loader:
                # Convert to numpy for sklearn
                X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
                y_batch = y_batch.to(self.device)
                mask = (y_batch != self.config.train.model.padding_index).to(self.device)
                
                # Normalization with scaler
                batch_size, seq_len, feat_dim = X_batch.shape
                X_flat = X_batch.reshape(-1, feat_dim)
                X_scaled = self.scaler.transform(X_flat)
                X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
                
                # Get logits from logistic regression
                X_flat = X_batch.reshape(-1, feat_dim)
                probs = self.pre_model.predict_proba(X_flat)
                logits = torch.from_numpy(probs).float()
                logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(self.device)
                
                # Loss
                log_likelihood = self.model(logits, y_batch, mask)
                loss = -log_likelihood.mean()
                
                # Get predictions
                preds = self.model.viterbi_decode(logits, mask) # type: ignore
                
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