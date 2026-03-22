import os, csv
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from Utils.eval_utils import load_aggregated, AggregatedData

OUTPUT_DIR=os.path.join(Path(__file__).resolve().parent, "EvalOutputs")
MODEL_DIR=os.path.join(Path(__file__).resolve().parent.parent, "Models")
MIR_LABELS=["Model", "Základní tón", "Dur-Moll", "Tercie", "Trojzvuk", "Septima", "Čtyřzvuk"]
TIME_LABELS=["Model", "Čas (s)"]
# MIR_LABELS=["Model", "Root", "MajMin", "Thirds", "Triads", "Sevenths", "Tetrads"]

def generate_total_table(*trios: tuple[str, str, bool], filename: str):
    data = []
    for model, name, crf in trios:
        model_path = os.path.join(MODEL_DIR, model)

        if not os.path.exists(model_path):
            print(f"Model {model} not found!")
            return
    
        # Load aggregation files
        if not crf:
            normal = os.path.join(model_path, "aggregated_data.json")
            aggregated = load_aggregated(normal)
        else:
            normal = os.path.join(model_path, "crf_aggregated_data.json")
            aggregated = load_aggregated(normal)

        # Prepare data 
        fields = [aggregated.root, aggregated.majmin, aggregated.thirds, aggregated.triads, aggregated.sevenths, aggregated.tetrads]
        data.append((name, *[round(np.mean(f)*100, 2) for f in fields]))

    #  Save table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MIR_LABELS)
        for row in data:
            writer.writerow(row)
    print(f"Saved to {out_path}")

def generate_times_table(*trios: tuple[str, str, bool], filename: str):
    data = []
    for model, name, crf in trios:
        model_path = os.path.join(MODEL_DIR, model)

        if not os.path.exists(model_path):
            print(f"Model {model} not found!")
            return
    
        # Load aggregation files
        if not crf:
            normal = os.path.join(model_path, "aggregated_data.json")
            aggregated = load_aggregated(normal)
        else:
            normal = os.path.join(model_path, "crf_aggregated_data.json")
            aggregated = load_aggregated(normal)

        # Prepare data 
        data.append((name ,np.mean(aggregated.inference_times)))

    #  Save table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TIME_LABELS)
        for row in data:
            writer.writerow(row)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    # generate_total_table(("log_pcp_complex", "Logistická regrese", True), ("cnn_144_complex", "CNN", True), ("cr2_144_complex", "CRNN (CR2)", False), ("btc_144_complex", "BTC", False), filename="avg_accuracies")
    # generate_total_table(("cnn_144_complex_multi", "CNN", True), ("cr2_144_complex_multi", "CRNN (CR2)", False), ("btc_144_complex_multi", "BTC", False), filename="multitask_accuracies")
    generate_total_table(("cr2_144_majmin_multi_final", "Majmin", False), ("cr2_144_majmin7_multi_final", "Majmin7", False), ("cr2_144_complex_multi_final", "Complex", False), filename="final_accuracies")
    # generate_times_table(("log_pcp_complex", "Logistická regrese", True), ("cnn_144_complex", "CNN", True), ("cr2_144_complex", "CRNN (CR2)", False), ("btc_144_complex", "BTC", False), filename="avg_infer_times")