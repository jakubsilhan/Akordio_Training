import os, shutil, joblib, time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Services.DatasetLoaderService import DatasetPathService
from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Tools.Chords import Chords, Complexity

def accuracy_fn(y_real, y_pred):
    mask = y_real != 0
    acc = (y_pred[mask] == y_real[mask]).mean() * 100
    return acc

class LogisticTrainer:
    """Trainer for Logistic Regression models using sklearn"""
    def __init__(self, config: Config):
        self.config = config
        self.loader = DatasetPathService(config)
        self.model_folder = os.path.join(
            config.train.model_path, 
            config.train.model_name, 
            str(config.train.test_fold)
        )
        self.chord_tool = Chords()
        self.complexity = self._get_complexity()

    def _get_complexity(self) -> Complexity:
        """
        Get chord complexity from config
        """
        match self.config.train.model_complexity:
            case "complex":
                return Complexity.COMPLEX
            case "majmin7":
                return Complexity.MAJMIN7
            case _:
                return Complexity.MAJMIN

    def train(self):
        """Logistic Regression training"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)

        # Load data
        train_paths, test_paths = self.loader.load_data_paths()

        train_X_list = []
        train_y_list = []
        for path in tqdm(train_paths, desc="Loading training data"):
            X_tensor, y_tensor = self._load_data(path)
            train_X_list.append(X_tensor)
            train_y_list.append(y_tensor)

        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        test_X_list = []
        test_y_list = []
        for path in tqdm(test_paths, desc="Loading test data"):
            X_tensor, y_tensor = self._load_data(path)
            test_X_list.append(X_tensor)
            test_y_list.append(y_tensor)

        test_X = np.concatenate(test_X_list, axis=0)
        test_y = np.concatenate(test_y_list, axis=0)

        # Normalization
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # Model
        start_time = time.time()
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=self.config.base.random_seed
        )

        # Training
        clf.fit(train_X, train_y)

        # Validation
        preds = clf.predict(test_X)

        # Accuracy
        acc = accuracy_fn(test_y, preds)

        # Logging
        print(f"Logistic Regression Accuracy: {acc}")

        # Saving model
        total_time = time.time() - start_time
        joblib.dump({"model": clf, "scaler": scaler, "time": total_time, "accuracy": acc}, os.path.join(self.model_folder,"model.joblib"))


    def _load_data(self, path: str):
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y_raw = data["y"]
        
        # Encode labels
        y = np.array([
            self.chord_tool.encode(
                chord=self.chord_tool.reduce(chord, self.complexity), 
                type=self.complexity
            ) for chord in y_raw
        ], dtype=np.int64)
        
        # Apply log transformation for CQT (not for PCP)
        if not self.config.data.preprocess.pcp.enabled:
            X = np.log(X + 1e-6)
        
        return X, y