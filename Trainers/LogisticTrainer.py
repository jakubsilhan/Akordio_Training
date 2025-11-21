import os, shutil, joblib
import numpy as np

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
            config.train.model_path, 
            config.train.model_name, 
            str(config.train.test_fold)
        )

    def train(self):
        """Logistic Regression training"""
        # Setup
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy2("config.yaml", self.model_folder)

        # Load data
        train_tensors, test_tensors = self.loader.load_data()

        train_x_list, train_y_list = zip(*train_tensors)
        train_X = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

        test_x_list, test_y_list = zip(*test_tensors)
        test_X = np.concatenate(test_x_list, axis=0)
        test_y = np.concatenate(test_y_list, axis=0)

        # train_X = np.array(train_X, dtype=np.float32)
        # train_y = np.array(train_y, dtype=np.int64)
        # test_X = np.array(test_X, dtype=np.float32)
        # test_y = np.array(test_y, dtype=np.int64)

        # Normalization
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # Model
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
        joblib.dump({"model": clf, "scaler": scaler}, os.path.join(self.model_folder,"model.joblib"))