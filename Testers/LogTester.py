import time, mir_eval, torch, os, joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Testers.BaseTester import BaseTester

class LogTester(BaseTester):
    """
    Tester class for sklearn chord recognition models
    """

    def test(self, test: bool = False) -> None:
        """Main testing loop"""
        # Load data
        if test:
            test_tensors = self.loader.load_test_data()
        else:
            test_tensors = self.loader.load_valid_data()
        test_dataloader = self.create_dataloader(test_tensors)

        # Load pre-model
        model_path = os.path.join(self.model_folder, "model.joblib")
        loaded = joblib.load(model_path)
        pre_model: LogisticRegression = loaded["model"]
        scaler: StandardScaler = loaded["scaler"]

        # Initializations
        evals = []
        times = []

        # Conf matrix preparation
        labels = self.chord_tool.get_labels(self.complexity)
        num_classes = len(labels)
        conf_m = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Batches
        for X_batch, y_batch in tqdm(test_dataloader, desc="Testing model"):
            # Move to device (GPU or CPU)
            X_batch = X_batch.detach().cpu().numpy().astype(np.float32)

            # Normalization
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            X_scaled = scaler.transform(X_flat)
            X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)

            targs = y_batch
            mask = (targs != self.config.train.model.padding_index)
            y_safe = y_batch.clone()
            y_safe[~mask] = 0

            # Forward pass
            X_flat = X_batch.reshape(-1, feat_dim)

            start_t = time.perf_counter()

            full_probs = np.zeros((X_flat.shape[0], self.config.train.model.output), dtype=np.float32)
            full_probs[:, pre_model.classes_] = pre_model.predict_proba(X_flat)
            preds_flat = np.argmax(full_probs, axis=1).reshape(batch_size, seq_len)

            end_t = time.perf_counter()
            times.append(end_t - start_t)

            # Conf matrix aggregation
            mask_flat = mask.view(-1).numpy().astype(bool)
            y_true = y_safe.view(-1).numpy()
            y_pred = preds_flat.reshape(-1)
            conf_m += confusion_matrix(y_true[mask_flat], y_pred[mask_flat], labels=range(num_classes))

            # Per fragment
            for i in range(batch_size):
                preds_sq = preds_flat[i].tolist()
                targs_sq = targs[i].tolist()

                predictions = [self.chord_tool.decode(chord, self.complexity) for chord in preds_sq]
                targets = [self.chord_tool.decode(chord, self.complexity) for chord in targs_sq]

                pred_int, pred_lab, targ_int, targ_lab = self.create_interval_sets(predictions, targets)

                pred_intervals = np.array(pred_int)
                targ_intervals = np.array(targ_int)

                evals.append(mir_eval.chord.evaluate(targ_intervals, targ_lab, pred_intervals, pred_lab))

        self.eval_data.confusion_matrix = conf_m.tolist()
        self.eval_data.conf_labels = labels

        # Aggregations
        self.process_results(evals, times)

        # Outputs
        self.save_results('evaluation.json')
        self.print_results()