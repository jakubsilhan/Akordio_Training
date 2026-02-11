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

    # Trainer specific setup
    def setup(self):
        torch.manual_seed(self.config.base.random_seed)
        # Pathing
        self.prefix = "crf_"

        # Data
        train_tensors, valid_tensors = self.loader.load_data()
        self.train_loader, self.val_loader = self.create_dataloaders(train_tensors, valid_tensors)

        # Pre-model
        self.pre_model = self.create_model()
        model_path = os.path.join(self.model_folder, "best_model.pt")
        loaded = torch.load(model_path, map_location=self.device)
        self.pre_model.load_state_dict(loaded['model'])
        normalization = loaded['normalization']
        self.train_mean = normalization['mean']
        self.train_std = normalization['std']

        self.pre_model.eval()
        for p in self.pre_model.parameters():
            p.requires_grad = False

        # Model
        self.model = CRF(num_labels=self.config.train.model.output).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.model.learning_rate)
        self.state, self.train_mean, self.train_std = self.load_checkpoint_if_exists(self.model, self.optimizer, self.train_mean, self.train_std, self.prefix)

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        self.model.train()
        losses = []
        accuracies = []
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mask = (y_batch != self.config.train.model.padding_index).to(self.device)
            
            # Normalization
            X_batch = (X_batch - self.train_mean) / self.train_std
            
            # Forward pass
            with torch.inference_mode():
                logits = self.pre_model(X_batch)
            preds = self.model.viterbi_decode(logits, mask)
            
            # Convert to tensor and pad
            preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
            preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=self.config.train.model.padding_index)

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
                mask = (y_batch != self.config.train.model.padding_index).to(self.device)
                
                # Normalization
                X_batch = (X_batch - self.train_mean) / self.train_std
                
                # Forward pass
                logits = self.pre_model(X_batch)

                #Loss and accuracy
                log_likelihood = self.model(logits, y_batch, mask)
                loss = -log_likelihood.mean()

                preds = self.model.viterbi_decode(logits, mask)
                
                # Convert to tensor and pad
                preds_tensor = [torch.tensor(p, device=self.device) for p in preds]
                preds_padded = pad_sequence(preds_tensor, batch_first=True, padding_value=self.config.train.model.padding_index)
                
                acc = accuracy_fn(y_batch, preds_padded, self.config.train.model.padding_index)
                
                losses.append(loss.cpu().item())
                accuracies.append(acc)
        
        average_loss = sum(losses)/len(losses)
        average_acc = sum(accuracies)/len(accuracies) 

        return average_loss, average_acc
