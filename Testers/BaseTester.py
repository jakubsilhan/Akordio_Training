import os, json, time, mir_eval, torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field, asdict
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Dict

from Akordio_Core.Tools.Chords import Chords, Complexity
from Akordio_Core.Classes.NetConfig import Config
from Akordio_Core.Classes.SongDataset import SongDataset, make_collate_fn
from Services.DatasetLoaderService import DatasetLoaderService

from Neural_Nets.CNN import Model as CNN
from Neural_Nets.CR1 import Model as CR1
from Neural_Nets.CR2 import Model as CR2
from Neural_Nets.BTC import Model as BTC

@dataclass
class EvalData:
    """Holds the evaluation data"""
    results: Dict = field(default_factory=dict)
    epoch_count: int = 0
    train_time_per_epoch: float = 0.0
    inference_time_per_batch: float = 0.0
    confusion_matrix: List = field(default_factory=list)
    conf_labels: List = field(default_factory=list)

class BaseTester:
    """
    Tester class for basic PyTorch chord recognition models
    """
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.complexity = self._get_complexity()
        self.chord_tool = Chords()
        self.loader = DatasetLoaderService(config)
        self.eval_data = EvalData()
        self.model_folder = os.path.join(
            Path(__file__).resolve().parent.parent,
            "Models",
            config.train.model_name, 
            str(config.train.val_fold)
        )

    def _get_complexity(self) -> Complexity:
        """Get chord complexity from config"""
        match self.config.train.model_complexity:
            case "complex":
                return Complexity.COMPLEX
            case "majmin7":
                return Complexity.MAJMIN7
            case _:
                return Complexity.MAJMIN

    def create_dataloader(self, test_tensors) -> DataLoader:
        """Create test dataloader"""
        test_dataset = SongDataset(test_tensors, self.config)

        collate_fn = make_collate_fn(self.config.train.model.padding_index)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.train.model.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        return test_dataloader
    
    def create_model(self) -> nn.Module:
        """Create and return the appropriate model based on config"""
        model_classes = {
            "BTC": BTC,
            "CR2": CR2,
            "CNN": CNN,
        }

        model_type = self.config.train.model_type
        model_class = model_classes.get(model_type, CR1) # Default to CR1
        
        # Initialize model
        model = model_class(config=self.config, device=self.device).to(self.device)
        return model
    
    def load_model_weights(self, model: nn.Module, prefix: str = "") -> Tuple[float, float]:
        """Loads model weights and data normalization values"""
        model_path = os.path.join(self.model_folder, f"{prefix}best_model.pt")
        loaded = torch.load(model_path, map_location=self.device)
        model.load_state_dict(loaded["model"], strict=False)
        
        normalization = loaded['normalization']

        self.eval_data.epoch_count = loaded['epoch']
        self.eval_data.train_time_per_epoch = loaded['total_time']/self.eval_data.epoch_count

        return normalization["mean"], normalization["std"]
    
    def test(self, test: bool = False) -> None:
        """Main testing loop"""

        # Load data
        if test:
            test_tensors = self.loader.load_test_data()
        else:
            test_tensors = self.loader.load_valid_data()
        test_dataloader = self.create_dataloader(test_tensors)

        # Create model
        model = self.create_model()

        # Load model and normalization
        norm_mean, norm_std = self.load_model_weights(model)

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

                # Forward pass
                if self.device == "cuda": torch.cuda.synchronize()
                start_t = time.perf_counter()
                logits = model(X_batch)
                if self.device == "cuda": torch.cuda.synchronize()
                end_t = time.perf_counter()
                times.append(end_t-start_t)

                preds = torch.softmax(logits, dim=2).argmax(dim=2)

                # Conf matrix aggregation
                y_true = y_batch.view(-1).cpu().numpy()
                y_pred = preds.view(-1).cpu().numpy()
                conf_m += confusion_matrix(y_true, y_pred, labels=range(num_classes))

                # Per fragment
                for i in range(X_batch.size(0)):

                    # Conversions
                    preds_sq = preds[i].detach().cpu().tolist()
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
        self.save_results('evaluation.json')
        self.print_results()


    def create_interval_sets(self, predictions: List[str], targets: List[str]) -> Tuple[List, List, List, List]:
        """Data aggregation for testing results"""
        # Initializations
        frame_duration = self.config.data.preprocess.hop_length/self.config.data.preprocess.sampling_rate 
        start_pred = 0
        start_targ = 0
        pred_int = []
        pred_lab = []
        targ_int = []
        targ_lab = []
        
        # Aggregating intervals
        for j in range(len(predictions)):
            # Pred intervals
            if j == 0:
                prev_chord_pred = predictions[j]
                prev_chord_targ = targets[j]
                continue

            if predictions[j] != prev_chord_pred:
                start_time = start_pred * frame_duration
                end_time = j * frame_duration
                pred_int.append([start_time, end_time])
                pred_lab.append(prev_chord_pred)
                prev_chord_pred = predictions[j]
                start_pred = j

            if targets[j] != prev_chord_targ:
                start_time = start_targ * frame_duration
                end_time = j * frame_duration
                targ_int.append([start_time, end_time])
                targ_lab.append(prev_chord_targ)
                prev_chord_targ = targets[j]
                start_targ = j

        # Final intervals
        pred_int.append([start_pred*frame_duration, len(predictions)*frame_duration])
        pred_lab.append(prev_chord_pred)

        targ_int.append([start_targ*frame_duration, len(targets)*frame_duration])
        targ_lab.append(prev_chord_targ)

        return pred_int, pred_lab, targ_int, targ_lab
    
    # TODO finish up
    # def save_preds(self, output_name: str, pred_int: List, pred_lab: List, targ_int: List, targ_lab: List) -> None:
    #     """Saves prediction results"""
    #     # Prepare dir
    #     output_dir = os.path.join("Test_Outputs", output_name)
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Save preds
    #     pred_str = ""
    #     for label, (start, end) in zip(pred_lab, pred_int):
    #         pred_str += f'{start:3f} {end:3f} {label} \n'

    #     with open(os.path.join(output_dir, "preds.lab"), "w") as file:
    #         file.write(pred_str)

    #     # Save targets
    #     targ_str = ""
    #     for label, (start, end) in zip(targ_lab, targ_int):
    #         targ_str += f'{start:3f} {end:3f} {label} \n'

    #     with open(os.path.join(output_dir, "targs.lab"), "w") as file:
    #         file.write(targ_str)

    def process_results(self, evaluations: List, times: List) -> Dict:
        """Processes aggregated data into averaged results"""
        if not evaluations or not times:
            print("No evaluations computed.")
            return {}
        
        # Metrics
        aggregated = defaultdict(float)
        for evaluation in evaluations:
            for k, v in evaluation.items():
                aggregated[k] += v
        average_eval = OrderedDict((k, aggregated[k] / len(evaluations)) for k in evaluations[0].keys())

        # Time
        avg_time = float(np.average(times))
        self.eval_data.results = average_eval
        self.eval_data.inference_time_per_batch = avg_time
        return average_eval
    
    def save_results(self, filename: str) -> None:
        """Saves results"""
        with open(os.path.join(self.model_folder, filename), 'w') as f:
            json.dump(asdict(self.eval_data), f, indent=4)

    def print_results(self) -> None:
        """Prints formatted results"""
        print("Testing results:")
        for k, v in self.eval_data.results.items():
            print(f"{k:>15}: {v:.4f}")