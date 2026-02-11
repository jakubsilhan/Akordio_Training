import os, torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Tools.Chords import Chords, Complexity

class DatasetLoaderService:
    """
    Data loading class for model training purposes
    """
    def __init__(self, config: Config):
        self.config = config
        self.chord_tool = Chords()
        self.complexity = self._get_complexity()

    def _get_complexity(self) -> Complexity:
        """Get chord complexity from config"""
        match self.config.train.model_complexity:
            case "complex":
                return Complexity.COMPLEX
            case "majmin7":
                return Complexity.MAJMIN7
            case _:
                return Complexity.MAJMIN

    def load_data(self, multitask: bool = False) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Load train and valid data tensors"""
        self.multitask = multitask
        train_tensors = self._load_train_data()
        valid_tensors = self.load_valid_data(multitask)
        return train_tensors, valid_tensors
    
    def _load_train_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load training data from all folds except valid fold"""
        train_tensors = []
        # TODO add the check for missing dataset and folds here
        data_path = os.path.join(self.config.train.data_source, "train")
        for fold in tqdm(os.listdir(data_path), desc="Loading train folds"):
            if fold == "config.yaml" or fold == str(self.config.train.val_fold): # Skip dataset config and test and val fold
                continue
            
            fold_dir = os.path.join(data_path, fold)
            train_tensors.extend(self._load_fold(fold_dir))
        
        return train_tensors
    
    def load_valid_data(self, multitask:bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load valid data from valid fold"""
        self.multitask = multitask
        valid_tensors = []
        valid_fold_path = os.path.join(
            self.config.train.data_source,
            "train",
            str(self.config.train.val_fold)
        )
        
        for fragment in tqdm(os.listdir(valid_fold_path), desc="Loading valid fold"):
            if not fragment.endswith(".npz") or "_shift00_" not in fragment: # Skip augmented data for validating
                continue
            
            fragment_path = os.path.join(valid_fold_path, fragment)
            tensor_pair = self._load_fragment(fragment_path)
            if tensor_pair:
                valid_tensors.append(tensor_pair)
        
        return valid_tensors
    
    def load_test_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load test data"""
        valid_tensors = []
        test_path = os.path.join(
            self.config.train.data_source,
            "test",
            "0"
        )
        
        for fragment in tqdm(os.listdir(test_path), desc="Loading test data"):
            if not fragment.endswith(".npz") or "_shift00_" not in fragment: # Skip augmented data for validating
                continue
            
            fragment_path = os.path.join(test_path, fragment)
            tensor_pair = self._load_fragment(fragment_path)
            if tensor_pair:
                valid_tensors.append(tensor_pair)
        
        return valid_tensors
    
    def _load_fold(self, fold_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load all fragments from a fold directory"""
        tensors = []
        for fragment in os.listdir(fold_dir):
            if not fragment.endswith(".npz"):
                continue
            
            fragment_path = os.path.join(fold_dir, fragment)
            tensor_pair = self._load_fragment(fragment_path)
            if tensor_pair:
                tensors.append(tensor_pair)
        
        return tensors
    
    def _load_fragment(self, fragment_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single fragment and encode chords"""
        data = np.load(fragment_path)
        X = data["X"]
        y_raw = data["y"]

        if self.multitask:
            y = [self.chord_tool.encode_multi(
                chord=self.chord_tool.reduce(chord, self.complexity),
                type=self.complexity
            ) for chord in y_raw]
        else:
            y = [self.chord_tool.encode(
                chord=self.chord_tool.reduce(chord, self.complexity), 
                type=self.complexity
            ) for chord in y_raw]
        
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return (X_tensor, y_tensor)
