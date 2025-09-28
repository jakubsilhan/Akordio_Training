from collections import defaultdict, OrderedDict
import os, json
import mir_eval
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.chords import Chords, Complexity
from core.net_config import Config, load_config
from core.song_dataset import SongDataset, make_collate_fn
from Neural_Nets.CR1 import Model as CR1
from Neural_Nets.SimpleLSTM import Model as SimpleLSTM
from Neural_Nets.BTC import Model as BTC

def test(config: Config):
    # Initialization
    model_folder = os.path.join(config.train.model_path, config.train.model_name, str(config.train.test_fold))

    chord_tool = Chords()

    match config.train.model_complexity:
        case "complex":
            complexity = Complexity.COMPLEX
        case "majmin7":
            complexity = Complexity.MAJMIN7
        case default:
            complexity = Complexity.MAJMIN

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading data
    fragments_song = defaultdict(list)
    test_fold_path = os.path.join(config.train.data_source, str(config.train.test_fold - 1))
    for fragment in tqdm(os.listdir(test_fold_path), desc="Loading test fold"):
        if not fragment.endswith(".npz"):
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
        fragments_song[song_name].append((X_tensor, y_tensor))

    match config.train.model_type:
        case "SimpleLSTM":
            model = SimpleLSTM(
                config=config,
                device=device
            ).to(device)
        case "BTC":
            model = BTC(
                config=config,
                device=device
            ).to(device)
        case default:
            model = CR1(
                config=config,
                device=device
            ).to(device)
    
    model.to(device)
    model_path = os.path.join(model_folder, "final_model.pt")
    loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(loaded['model'])
    normalization = loaded['normalization']

    evals = []

    for name, tensors in fragments_song.items():
        song_dataset = SongDataset(tensors)
        song_dataloader = DataLoader(song_dataset, batch_size=8, shuffle=False, collate_fn=make_collate_fn(config.train.model.padding_index))

        # Initializations
        start_pred = 0
        start_targ = 0
        frame_duration = config.data.preprocess.hop_length/config.data.preprocess.sampling_rate

        pred_int = []
        pred_lab = []
        targ_int = []
        targ_lab = []


        ### Testing
        predictions = []
        targets = []

        model.eval()

        for X_batch, y_batch in song_dataloader:
            #### Move to device (GPU or CPU)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #### Normalization
            X_batch = (X_batch-normalization['mean'])/normalization['std']
            targs = y_batch

            with torch.inference_mode():
                #### 1. Forward pass
                logits = model(X_batch)
                preds = torch.softmax(logits, dim=2).argmax(dim=2)

                predictions.extend(preds.view(-1).detach().cpu())
                targets.extend(targs.view(-1).detach().cpu())
        
        predictions = [chord_tool.decode(chord, complexity) for chord in predictions]
        targets = [chord_tool.decode(chord, complexity) for  chord in targets]

        # Aggregating intervals
        for i in range(len(predictions)):
            # Pred intervals
            if i == 0:
                prev_chord_pred = predictions[i]
                prev_chord_targ = targets[i]
                continue

            if predictions[i] != prev_chord_pred:
                start_time = start_pred * frame_duration
                end_time = i * frame_duration
                pred_int.append([start_time, end_time])
                pred_lab.append(prev_chord_pred)
                prev_chord_pred = predictions[i]
                start_pred = i

            if targets[i] != prev_chord_targ:
                start_time = start_targ * frame_duration
                end_time = i * frame_duration
                targ_int.append([start_time, end_time])
                targ_lab.append(prev_chord_targ)
                prev_chord_targ = targets[i]
                start_targ = i

        # Final intervals
        pred_int.append([start_pred*frame_duration, len(predictions)*frame_duration])
        pred_lab.append(prev_chord_pred)

        targ_int.append([start_targ*frame_duration, len(targets)*frame_duration])
        targ_lab.append(prev_chord_targ)

        output_dir = os.path.join("Test_Outputs", name)
        os.makedirs(output_dir, exist_ok=True)
        # Save preds
        pred_str = ""
        for label, (start, end) in zip(pred_lab, pred_int):
            pred_str += f'{start:3f} {end:3f} {label} \n'

        with open(os.path.join(output_dir, "preds.lab"), "w") as file:
            file.write(pred_str)

        # Save targets
        targ_str = ""
        for label, (start, end) in zip(targ_lab, targ_int):
            targ_str += f'{start:3f} {end:3f} {label} \n'

        with open(os.path.join(output_dir, "targs.lab"), "w") as file:
            file.write(targ_str)

            
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