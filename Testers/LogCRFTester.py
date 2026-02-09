import time, mir_eval, torch, os, joblib
import numpy as np

from tqdm import tqdm

from TorchCRF import CRF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Testers.BaseTester import BaseTester

class LogCRFTester(BaseTester):
    """
    Tester class for sklearn chord recognition models with incorporated linear chain CRF
    """

    def test(self) -> None:
        """Main testing loop"""
        # Load data
        test_tensors = self.loader.load_valid_data()
        test_dataloader = self.create_dataloader(test_tensors)

        # Load pre-model
        model_path = os.path.join(self.model_folder, "model.joblib")
        loaded = joblib.load(model_path)
        pre_model: LogisticRegression = loaded["model"]
        scaler: StandardScaler = loaded["scaler"]

        # Create CRF
        crf = CRF(num_labels=self.config.train.model.output).to(self.device)

        # Load CRF
        _, _ = self.load_model_weights(crf, "crf_")

        # Initializations
        evals = []
        times = []

        with torch.inference_mode():
            # Batches
            for X_batch, y_batch in tqdm(test_dataloader, desc="Testing model"):
                # Move to device (GPU or CPU)
                X_batch = X_batch.detach().cpu().numpy().astype(np.float32)
                y_batch = y_batch.to(self.device)

                # Normalization
                batch_size, seq_len, feat_dim = X_batch.shape
                X_flat = X_batch.reshape(-1, feat_dim)
                X_scaled = scaler.transform(X_flat)
                X_batch = X_scaled.reshape(batch_size, seq_len, feat_dim)
                targs = y_batch
                mask = (targs != self.config.train.model.padding_index).to(self.device)

                # Forward pass
                batch_size, seq_len, feat_dim = X_batch.shape
                X_flat = X_batch.reshape(-1, feat_dim)

                if self.device == "cuda": torch.cuda.synchronize()
                start_t = time.perf_counter()

                probs = pre_model.predict_proba(X_flat)
                logits = torch.from_numpy(probs).float()
                logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(self.device)
                preds = crf.viterbi_decode(logits, mask) # type: ignore

                if self.device == "cuda": torch.cuda.synchronize()
                end_t = time.perf_counter()
                times.append(end_t-start_t)

                # Per fragment
                for i in range(X_batch.size(0)):

                    # Conversions
                    preds_sq = preds[i]
                    targs_sq = targs[i].detach().cpu().tolist()

                    predictions = [self.chord_tool.decode(chord, self.complexity) for chord in preds_sq]
                    targets = [self.chord_tool.decode(chord, self.complexity) for  chord in targs_sq]

                    pred_int, pred_lab, targ_int, targ_lab = self.create_interval_sets(predictions, targets)

                    # Numpy conversion
                    pred_intervals = np.array(pred_int)
                    targ_intervals = np.array(targ_int)

                    # Mir evals
                    evals.append(mir_eval.chord.evaluate(targ_intervals, targ_lab, pred_intervals, pred_lab))

        # Aggregations
        results = self.process_results(evals, times)

        # Outputs
        self.save_results(results, 'crf_test_mir_eval.json')
        self.print_results(results)