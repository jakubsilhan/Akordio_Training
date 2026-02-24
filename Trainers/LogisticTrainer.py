import os, shutil, joblib, time
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Services.DatasetLoaderService import DatasetLoaderService

def accuracy_fn(y_real, y_pred):
    mask = y_real != 0
    acc = (y_pred[mask] == y_real[mask]).mean() * 100
    return acc

class LogisticTrainer:
    """Trainer for Logistic Regression models using sklearn"""
    def __init__(self, config):
        self.config = config
        self.loader = DatasetLoaderService(config)
        self.model_folder = os.path.join(
            Path(__file__).resolve().parent.parent,
            "Models",
            config.train.model_name, 
            str(config.train.val_fold)
        )

    def train(self):
        """Logistic Regression training"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)

        # Load data
        train_tensors, valid_tensors = self.loader.load_data()

        train_x_list, train_y_list = zip(*train_tensors)
        train_X = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        valid_x_list, valid_y_list = zip(*valid_tensors)
        valid_X = np.concatenate(valid_x_list, axis=0)
        valid_y = np.concatenate(valid_y_list, axis=0)

        # Normalization
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        valid_X = scaler.transform(valid_X)

        # Model
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            # class_weight="balanced",
            random_state=self.config.base.random_seed
        )

        start_time = time.time()
        # Training
        clf.fit(train_X, train_y)
        # Validation
        preds = clf.predict(valid_X)
        total_time = time.time() - start_time

        # Accuracy
        acc = accuracy_fn(valid_y, preds)

        # Logging
        print(f"Logistic Regression Accuracy: {acc}")

        # Saving model
        joblib.dump({"model": clf, "scaler": scaler, "total_time": total_time}, os.path.join(self.model_folder,"model.joblib"))

    def train_final(self, epoch_count: int):
        """Logistic Regression final training"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)

        # Load data
        train_tensors, _ = self.loader.load_data()

        train_x_list, train_y_list = zip(*train_tensors)
        train_X = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        # Normalization
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)

        # Model
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=epoch_count,
            random_state=self.config.base.random_seed
        )

        start_time = time.time()
        # Training
        clf.fit(train_X, train_y)
        total_time = time.time() - start_time

        # Saving model
        joblib.dump({"model": clf, "scaler": scaler, "total_time": total_time}, os.path.join(self.model_folder,"model.joblib"))
