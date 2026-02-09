import os, shutil, torch, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple

from Utils.training_utils import accuracy_fn, adjusting_learning_rate, compute_mean_std
from Trainers.BaseTrainer import BaseTrainer

W_CHORD=1.0
W_ROOT=1.0
W_QUALITY=1.0

class MultiTrainer(BaseTrainer):
    """
    Training class for basic PyTorch chord recognition models utilizing multitask training
    """
    def train(self):
        """Main training loop"""
        # Setup
        model_name = self.config.train.model_name
        if not model_name.endswith("_multi"):
            model_name += "_multi"
        self.model_folder = os.path.join(
            self.config.train.model_path, 
            self.config.train.model_name+"_multi", 
            str(self.config.train.val_fold)
        )
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)
        torch.manual_seed(self.config.base.random_seed)
        
        # Load data
        train_tensors, valid_tensors = self.loader.load_data(multitask=True)
        train_dataloader, valid_dataloader = self.create_dataloaders(train_tensors, valid_tensors)
        
        # Compute normalization
        train_mean, train_std = compute_mean_std(train_dataloader)
        
        # Create model, loss, optimizer
        model = self.create_model()
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.train.model.padding_index)
        optimizer = optim.Adam(model.parameters(), lr=self.config.train.model.learning_rate, weight_decay=self.config.train.model.weight_decay, betas=(0.9, 0.98), eps=1e-9)
        
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
                train_loss, train_acc, train_root_acc, train_quality_acc = self.train_epoch(
                    model, train_dataloader, loss_fn, optimizer, train_mean, train_std
                )
                
                # Evaluate
                valid_loss, valid_acc, valid_root_acc, valid_quality_acc = self.evaluate_epoch(
                    model, valid_dataloader, loss_fn, train_mean, train_std
                )
                
                # Update state
                state.train_loss_list.append(train_loss)
                state.train_accuracy_list.append(train_acc)
                state.valid_loss_list.append(valid_loss)
                state.valid_accuracy_list.append(valid_acc)
                
                # Log progress
                tqdm.write(
                    f"Epoch: {epoch} | Loss: {train_loss:.4f} | "
                    f"Train [C: {train_acc:.1f}% R: {train_root_acc:.1f}% Q: {train_quality_acc:.1f}%] | "
                    f"Val [C: {valid_acc:.1f}% R: {valid_root_acc:.1f}% Q: {valid_quality_acc:.1f}%]"
                )
                
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

    def train_epoch(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, train_mean: float, train_std: float) -> Tuple[float, float, float, float]:
        """Train for one epoch and return average loss and accuracy"""
        model.train()
        losses = []
        losses, chord_accs, root_accs, quality_accs = [], [], [], []
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Normalization
            X_batch = (X_batch - train_mean) / train_std
            
            # Forward pass
            logits, root_logits, quality_logits = model.forward_multitask(X_batch) # type: ignore
            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            root_preds = torch.softmax(root_logits, dim=2).argmax(dim=2)
            quality_preds = torch.softmax(quality_logits, dim=2).argmax(dim=2)
            
            # Loss and accuracy            
            loss_chord = loss_fn(logits.view(-1, logits.size(-1)), y_batch[:, :, 0].reshape(-1))
            loss_root  = loss_fn(root_logits.view(-1, root_logits.size(-1)), y_batch[:, :, 1].reshape(-1))
            loss_qual  = loss_fn(quality_logits.view(-1, quality_logits.size(-1)), y_batch[:, :, 2].reshape(-1))
            
            total_loss = (W_CHORD*loss_chord) + (W_ROOT*loss_root) + (W_QUALITY*loss_qual)
            acc = accuracy_fn(y_batch[:,:,0], preds, self.config.train.model.padding_index)
            root_acc = accuracy_fn(y_batch[:,:,1], root_preds, self.config.train.model.padding_index)
            quality_acc = accuracy_fn(y_batch[:,:,2], quality_preds, self.config.train.model.padding_index)

            losses.append(total_loss.detach().cpu().item())
            chord_accs.append(acc)
            root_accs.append(root_acc)
            quality_accs.append(quality_acc)
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        average_loss = sum(losses)/len(losses)
        average_acc = sum(chord_accs)/len(chord_accs) 
        average_root_acc = sum(root_accs)/len(root_accs)
        average_quality_acc = sum(quality_accs)/len(quality_accs)

        return average_loss, average_acc, average_root_acc, average_quality_acc
    
    def evaluate_epoch(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, train_mean: float, train_std: float) -> Tuple[float, float, float, float]:
        """Evaluate model and return average loss and accuracy"""
        model.eval()
        losses = []
        losses, chord_accs, root_accs, quality_accs = [], [], [], []
        
        with torch.inference_mode():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Normalization
                X_batch = (X_batch - train_mean) / train_std
                
                # Forward pass
                logits, root_logits, quality_logits = model.forward_multitask(X_batch)  # type: ignore
                preds = torch.softmax(logits, dim=2).argmax(dim=2)
                root_preds = torch.softmax(root_logits, dim=2).argmax(dim=2)
                quality_preds = torch.softmax(quality_logits, dim=2).argmax(dim=2)
                
                # Loss and accuracy
                loss_chord = loss_fn(logits.view(-1, logits.size(-1)), y_batch[:, :, 0].reshape(-1))
                loss_root  = loss_fn(root_logits.view(-1, root_logits.size(-1)), y_batch[:, :, 1].reshape(-1))
                loss_qual  = loss_fn(quality_logits.view(-1, quality_logits.size(-1)), y_batch[:, :, 2].reshape(-1))
                
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
        average_root_acc = sum(root_accs)/len(root_accs)
        average_quality_acc = sum(quality_accs)/len(quality_accs)

        return average_loss, average_acc, average_root_acc, average_quality_acc