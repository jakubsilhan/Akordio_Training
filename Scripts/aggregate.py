import argparse, os, json
from pathlib import Path
from Utils.eval_utils import AggregatedData, save_aggregated


def agregate(model_path: str, prefix: str = ""):
    aggregated = AggregatedData()
    for fold in os.listdir(model_path):
        if "json" in fold: continue
        # Load
        file_path = os.path.join(model_path, fold, f"{prefix}evaluation.json")
        if not os.path.exists(file_path):
            print(f"Evaluation file for fold {fold} not found!")
            continue
        with open(file_path, mode="r", encoding="utf-8") as f:
            fold_aggregate = json.load(f)
        # Parse
        aggregated.epoch_counts.append(fold_aggregate["epoch_count"])
        aggregated.train_times.append(fold_aggregate["train_time_per_epoch"])
        aggregated.inference_times.append(fold_aggregate["inference_time_per_batch"])
        metrics = fold_aggregate["results"]
        aggregated.root.append(metrics["root"])
        aggregated.thirds.append(metrics["thirds"])
        aggregated.sevenths.append(metrics["sevenths"])
        aggregated.triads.append(metrics["triads"])
        aggregated.tetrads.append(metrics["tetrads"])
        aggregated.segmentation.append(metrics["seg"])
        aggregated.majmin.append(metrics["majmin"])      

    # Save to file
    save_aggregated(os.path.join(model_path, f"{prefix}aggregated_data.json"), aggregated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Aggrregator", description="Program for aggregating testing results")
    parser.add_argument("-m", "--model", required=True, type=str, help="Name of the model to aggregate")
    parser.add_argument("-c", "--crf", action="store_true", help="Whether to check the crf data")
    args = parser.parse_args()

    # Params
    model_name: str = args.model
    crf_enabled: bool = args.crf

    # Find model dir
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = os.path.join(project_root, "Models", model_name)
    if not os.path.exists(model_dir):
        print("Model not found")
        exit()

    # Data aggregation
    if not crf_enabled:
        agregate(model_dir)
    else:
        agregate(model_dir, "crf_")
