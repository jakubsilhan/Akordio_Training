import time, mir_eval, torch
import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from TorchCRF import CRF
from Testers.BaseTester import BaseTester


class CRFTester(BaseTester):
    """
    Tester class for PyTorch chord recognition models with incorporated linear chain CRF
    """

    def test(self, test: bool = False) -> None:
        """Main testing loop"""

        # Load data
        if test:
            test_tensors = self.loader.load_test_data()
        else:
            test_tensors = self.loader.load_valid_data()
        test_dataloader = self.create_dataloader(test_tensors)

        # Create pre_model
        pre_model = self.create_model()

        # Load pre_model and normalization
        norm_mean, norm_std = self.load_model_weights(pre_model)

        # Create CRF
        crf = CRF(num_labels=self.config.train.model.output).to(self.device)

        # Load CRF
        norm_mean, norm_std = self.load_model_weights(crf, "crf_")

        # Initializations
        evals = []
        times = []

        # Conf matrix preparation
        labels = self.chord_tool.get_labels(self.complexity) 
        num_classes = len(labels)
        conf_m = np.zeros((num_classes, num_classes), dtype=np.int64)

        with torch.inference_mode():
            # Batches
            for X_batch, y_batch in tqdm(test_dataloader, desc="Testing model"):
                # Move to device (GPU or CPU)
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Normalization
                X_batch = (X_batch-norm_mean)/norm_std
                targs = y_batch
                mask = (targs != self.config.train.model.padding_index).to(self.device)

                # Forward pass
                if self.device == "cuda": torch.cuda.synchronize()
                start_t = time.perf_counter()
                
                logits = pre_model(X_batch)
                preds = crf.viterbi_decode(logits, mask)
                
                if self.device == "cuda": torch.cuda.synchronize()
                end_t = time.perf_counter()
                times.append(end_t-start_t)

                # Conf matrix aggregation
                y_true = y_batch.view(-1).cpu().numpy()
                y_pred = np.array([label for seq in preds for label in seq])
                conf_m += confusion_matrix(y_true, y_pred, labels=range(num_classes))

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

        self.eval_data.confusion_matrix = conf_m.tolist()
        self.eval_data.conf_labels = labels

        # Aggregations
        self.process_results(evals, times)

        # Outputs
        self.save_results('crf_evaluation.json')
        self.print_results()