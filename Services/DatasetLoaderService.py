import os, torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Tools.Chords import Chords, Complexity

class DatasetPathService:
    def __init__(self, config: Config):
        self.config = config

    def load_data_paths(self) -> Tuple[List[str], List[str]]:
        """Load train and test data paths"""
        train_paths = self.load_train_paths()
        test_paths = self.load_test_paths()
        return train_paths, test_paths
    
    def load_train_paths(self) -> List[str]:
        """Load training paths from all folds except test fold"""
        train_paths = []
        # TODO add the check for missing dataset and folds here
        for fold in tqdm(os.listdir(self.config.train.data_source), desc="Loading train paths"):
            if fold == "config.yaml" or fold == str(self.config.train.test_fold): # Skip dataset config and training fold
                continue
            fold_dir = os.path.join(self.config.train.data_source, fold)
            train_paths.extend(self._load_fold(fold_dir))
        
        return train_paths
    
    def load_test_paths(self) -> List[str]:
        """Load test paths from test fold"""
        test_paths = []
        test_fold_path = os.path.join(
            self.config.train.data_source, 
            str(self.config.train.test_fold)
        )
        
        for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test paths"):
            if not fragment.endswith(".npz") or "_shift00_" not in fragment: # Skip augmented data for testing
                continue
            
            fragment_path = os.path.join(test_fold_path, fragment)
            test_paths.append(fragment_path)
        
        return test_paths
    
    def _load_fold(self, fold_dir: str) -> List[str]:
        """Load all fragment paths from a fold directory"""
        fragment_paths = []
        for fragment in os.listdir(fold_dir):
            if not fragment.endswith(".npz"):
                continue
            
            fragment_path = os.path.join(fold_dir, fragment)
            fragment_paths.append(fragment_path)
        
        return fragment_paths
