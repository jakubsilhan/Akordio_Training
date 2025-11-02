import os, pickle, shutil, joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from Akordio_Core.net_config import Config, load_config
from Akordio_Core.chords import Chords, Complexity

def accuracy_fn(y_real, y_pred, padding_index):
    mask = y_real != 0
    acc = (y_pred[mask] == y_real[mask]).mean() * 100
    return acc


def train(config: Config):
    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name, str(config.train.test_fold))
    os.makedirs(model_folder, exist_ok=True)
    shutil.copy2("config.yaml", model_folder)

    match config.train.model_complexity:
        case "complex":
            complexity = Complexity.COMPLEX
        case "majmin7":
            complexity = Complexity.MAJMIN7
        case _:
            complexity = Complexity.MAJMIN

    chord_tool = Chords()

    # Load data
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []

    ## Check for dataset
    # TODO add the check here

    ## Load Train
    # TODO add checks for missing folds
    for fold in tqdm(os.listdir(config.train.data_source), desc="Loading train folds"):
        if fold == "config.yaml":
            continue
        if fold == str(config.train.test_fold - 1):
            continue

        fold_dir = os.path.join(config.train.data_source, fold)
        for fragment in os.listdir(fold_dir):
            if not fragment.endswith(".npz"):
                continue
            fragment_path = os.path.join(fold_dir, fragment)
            data = np.load(fragment_path)
            X = data["X"]
            y_raw = data["y"]

            y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
                for chord in y_raw]

            train_x_list.extend(X)
            train_y_list.extend(y)

    ## Load Test
    test_fold_path = os.path.join(config.train.data_source, str(config.train.test_fold - 1))
    for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
        if not fragment.endswith(".npz"):
            continue

        if "_shift00_" not in fragment:
            continue

        fragment_path = os.path.join(test_fold_path, fragment)
        data = np.load(fragment_path)
        X = data["X"]
        y_raw = data["y"]

        y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
            for chord in y_raw]

        test_x_list.extend(X)
        test_y_list.extend(y)

    train_X = np.array(train_x_list, dtype=np.float32)
    train_y = np.array(train_y_list, dtype=np.int64)
    test_X = np.array(test_x_list, dtype=np.float32)
    test_y = np.array(test_y_list, dtype=np.int64)

    # Normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # Model
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
         random_state=config.base.random_seed
    )

    clf.fit(train_X, train_y)

    preds = clf.predict(test_X)

    acc = accuracy_fn(test_y, preds, 0)

    print(f"Logistic Regression Accuracy: {acc}")

    joblib.dump({"model": clf, "scaler": scaler}, os.path.join(model_folder,"model.joblib"))


if __name__=="__main__":
    config = load_config("config.yaml")
    train(config)