import os, torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Tools.Chords import Chords, Complexity

class DatasetLoaderService:
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

    def load_data(self) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Load train and test data tensors"""
        train_tensors = self._load_train_data()
        test_tensors = self._load_test_data()
        return train_tensors, test_tensors
    
    def _load_train_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load training data from all folds except test fold"""
        train_tensors = []
        # TODO add the check for missing dataset and folds here
        for fold in tqdm(os.listdir(self.config.train.data_source), desc="Loading train folds"):
            if fold == "config.yaml" or fold == str(self.config.train.test_fold - 1): # Skip dataset config and training fold
                continue
            
            fold_dir = os.path.join(self.config.train.data_source, fold)
            train_tensors.extend(self._load_fold(fold_dir))
        
        return train_tensors
    
    def _load_test_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load test data from test fold"""
        test_tensors = []
        test_fold_path = os.path.join(
            self.config.train.data_source, 
            str(self.config.train.test_fold - 1)
        )
        
        for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
            if not fragment.endswith(".npz") or "_shift00_" not in fragment: # Skip augmented data for testing
                continue
            
            fragment_path = os.path.join(test_fold_path, fragment)
            tensor_pair = self._load_fragment(fragment_path)
            if tensor_pair:
                test_tensors.append(tensor_pair)
        
        return test_tensors
    
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
        
        y = [self.chord_tool.encode(
            chord=self.chord_tool.reduce(chord, self.complexity), 
            type=self.complexity
        ) for chord in y_raw]
        
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return (X_tensor, y_tensor)
