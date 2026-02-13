import os, shutil, torch, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple

from Akordio_Core.Classes.NetConfig import Config
from Utils.training_utils import accuracy_fn, compute_mean_std
from Trainers.BaseTrainer import BaseTrainer

W_CHORD=1.0
W_ROOT=1.0
W_QUALITY=1.0

class MultiTrainer(BaseTrainer):
    """
    Training class for basic PyTorch chord recognition models utilizing multitask training
    """
    def __init__(self, config: Config):
        super().__init__(config)
        model_name = self.config.train.model_name
        if not model_name.endswith("_multi"):
            model_name += "_multi"
        self.model_folder = os.path.join(
            self.config.train.model_path, 
            self.config.train.model_name+"_multi", 
            str(self.config.train.val_fold)
        )


    def setup(self):
        torch.manual_seed(self.config.base.random_seed)
        # Pathing
        self.prefix = ""

        # Data
        train_tensors, valid_tensors = self.loader.load_data(multitask=True)
        self.train_loader, self.val_loader = self.create_dataloaders(train_tensors, valid_tensors)
        self.train_mean, self.train_std = compute_mean_std(self.train_loader)

        # Model
        self.model = self.create_model()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.train.model.padding_index)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.model.learning_rate, weight_decay=self.config.train.model.weight_decay)
        self.state, self.train_mean, self.train_std = self.load_checkpoint_if_exists(self.model, self.optimizer, self.train_mean, self.train_std)

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy"""
        self.model.train()
        losses, chord_accs, root_accs, quality_accs = [], [], [], []
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Normalization
            X_batch = (X_batch - self.train_mean) / self.train_std
            
            # Forward pass
            logits, root_logits, quality_logits = self.model.forward_multitask(X_batch) # type: ignore
            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            root_preds = torch.softmax(root_logits, dim=2).argmax(dim=2)
            quality_preds = torch.softmax(quality_logits, dim=2).argmax(dim=2)
            
            # Loss and accuracy            
            loss_chord = self.loss_fn(logits.view(-1, logits.size(-1)), y_batch[:, :, 0].reshape(-1))
            loss_root  = self.loss_fn(root_logits.view(-1, root_logits.size(-1)), y_batch[:, :, 1].reshape(-1))
            loss_qual  = self.loss_fn(quality_logits.view(-1, quality_logits.size(-1)), y_batch[:, :, 2].reshape(-1))
            
            total_loss = (W_CHORD*loss_chord) + (W_ROOT*loss_root) + (W_QUALITY*loss_qual)
            acc = accuracy_fn(y_batch[:,:,0], preds, self.config.train.model.padding_index)
            root_acc = accuracy_fn(y_batch[:,:,1], root_preds, self.config.train.model.padding_index)
            quality_acc = accuracy_fn(y_batch[:,:,2], quality_preds, self.config.train.model.padding_index)

            losses.append(total_loss.detach().cpu().item())
            chord_accs.append(acc)
            root_accs.append(root_acc)
            quality_accs.append(quality_acc)
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        average_loss = sum(losses)/len(losses)
        average_acc = sum(chord_accs)/len(chord_accs) 

        return average_loss, average_acc
    
    def evaluate_epoch(self) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy"""
        self.model.eval()
        losses, chord_accs, root_accs, quality_accs = [], [], [], []
        
        with torch.inference_mode():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Normalization
                X_batch = (X_batch - self.train_mean) / self.train_std
                
                # Forward pass
                logits, root_logits, quality_logits = self.model.forward_multitask(X_batch)  # type: ignore
                preds = torch.softmax(logits, dim=2).argmax(dim=2)
                root_preds = torch.softmax(root_logits, dim=2).argmax(dim=2)
                quality_preds = torch.softmax(quality_logits, dim=2).argmax(dim=2)
                
                # Loss and accuracy
                loss_chord = self.loss_fn(logits.view(-1, logits.size(-1)), y_batch[:, :, 0].reshape(-1))
                loss_root  = self.loss_fn(root_logits.view(-1, root_logits.size(-1)), y_batch[:, :, 1].reshape(-1))
                loss_qual  = self.loss_fn(quality_logits.view(-1, quality_logits.size(-1)), y_batch[:, :, 2].reshape(-1))
                
                total_loss = (W_CHORD * loss_chord) + (W_ROOT * loss_root) + (W_QUALITY * loss_qual)
                acc = accuracy_fn(y_batch[:,:,0], preds, self.config.train.model.padding_index)
                root_acc = accuracy_fn(y_batch[:,:,1], root_preds, self.config.train.model.padding_index)
                quality_acc = accuracy_fn(y_batch[:,:,2], quality_preds, self.config.train.model.padding_index)
                
                losses.append(total_loss.detach().cpu().item())
                chord_accs.append(acc)
                root_accs.append(root_acc)
                quality_accs.append(quality_acc)
        
        average_loss = sum(losses)/len(losses)
        average_acc = sum(chord_accs)/len(chord_accs) 

        return average_loss, average_acc