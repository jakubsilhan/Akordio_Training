from collections import defaultdict, OrderedDict
import os, json, joblib
import mir_eval
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Akordio_Core.chords import Chords, Complexity
from Akordio_Core.net_config import Config, load_config
from Akordio_Core.song_dataset import SongDataset, make_collate_fn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from TorchCRF import CRF

def test(config: Config):
    """
    Testing for a CRF model

    Uses the same configuration as the used recognition model.
    """

    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name, str(config.train.test_fold))

    chord_tool = Chords()

    match config.train.model_complexity:
        case "complex":
            complexity = Complexity.COMPLEX
        case "majmin7":
            complexity = Complexity.MAJMIN7
        case _:
            complexity = Complexity.MAJMIN

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading data
    fragments = list()
    test_fold_path = os.path.join(config.train.data_source, str(config.train.test_fold - 1))
    for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
        if not fragment.endswith(".npz"):
            continue

        if "_shift00_" not in fragment:
            continue

        fragment_path = os.path.join(test_fold_path, fragment)
        data = np.load(fragment_path)
        song_name = fragment.split("frag")[0]
        X = data["X"]
        y_raw = data["y"]

        y = [chord_tool.encode(chord=chord_tool.reduce(chord, complexity), type=complexity)
            for chord in y_raw]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        fragments.append((X_tensor, y_tensor))

    # Load pre-model
    model_path = os.path.join(model_folder, "model.joblib")
    loaded = joblib.load(model_path)
    pre_model: LogisticRegression = loaded["model"]
    scaler: StandardScaler = loaded["scaler"]

    crf = CRF(num_labels=config.train.model.output).to(device)
    crf_path = os.path.join(model_folder, "final_crf_model.pt")
    loaded_crf = torch.load(crf_path, map_location=device)
    crf.load_state_dict(loaded_crf['model'])

    evals = []

    test_dataset = SongDataset(fragments, config)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=make_collate_fn(config.train.model.padding_index))

    # Initializations
    frame_duration = config.data.preprocess.hop_length/config.data.preprocess.sampling_rate

    ### Testing

    for X_batch, y_batch in tqdm(test_dataloader, desc="Evaluating"):
        #### Move to device (GPU or CPU)
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = y_batch.to(device)
        mask = (y_batch != config.train.model.padding_index).to(device)

        #### Normalization
        X_batch = scaler.transform(X_batch)
        targs = y_batch

        with torch.inference_mode():
            batch_size, seq_len, feat_dim = X_batch.shape
            X_flat = X_batch.reshape(-1, feat_dim)
            probs = pre_model.predict_proba(X_flat)
            logits = torch.from_numpy(probs).float()
            logits = torch.log(logits + 1e-8).view(batch_size, seq_len, -1).to(device)
            preds = crf.viterbi_decode(logits, mask)  # type: ignore

    
        for i in range(len(X_batch)):  # Per fragment evaluation
            # Initializations 
            start_pred = 0
            start_targ = 0
            pred_int = []
            pred_lab = []
            targ_int = []
            targ_lab = []

            # Conversions
            preds_sq = preds[i]
            targs_sq = targs[i].detach().cpu().tolist()

            predictions = [chord_tool.decode(chord, complexity) for chord in preds_sq]
            targets = [chord_tool.decode(chord, complexity) for  chord in targs_sq]

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
            
            # Numpy conversion
            pred_intervals = np.array(pred_int)
            targ_intervals = np.array(targ_int)

            evals.append(mir_eval.chord.evaluate(targ_intervals, targ_lab, pred_intervals, pred_lab))

    # Accumulate sums
    aggregated = defaultdict(float)
    for song_eval in evals:
        for k, v in song_eval.items():
            aggregated[k] += v

    # Average
    average_eval = OrderedDict((k, aggregated[k] / len(evals)) for k in evals[0].keys())

    # Save to file
    with open(os.path.join(model_folder, 'test_mir_eval.json'), 'w') as f:
        f.write(json.dumps(average_eval))

    print("\n Average chord evaluation metrics:")
    for k, v in average_eval.items():
        print(f"{k:>15}: {v:.4f}")

if __name__ == "__main__":
    config = load_config("config.yaml")
    test(config)