import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from Utils.eval_utils import load_aggregated, AggregatedData

OUTPUT_DIR=os.path.join(Path(__file__).resolve().parent, "EvalOutputs")
MODEL_DIR=os.path.join(Path(__file__).resolve().parent.parent, "Models")
MIR_LABELS=["Základní tón", "Dur-Moll", "Tercie", "Trojzvuk", "Septima", "Čtyřzvuk"]
# MIR_LABELS=["Root", "MajMin", "Thirds", "Triads", "Sevenths", "Tetrads"]
WIDTH=0.35

def to_boxplot_data(agg: AggregatedData):
    fields = [agg.root, agg.majmin, agg.thirds, agg.triads, agg.sevenths, agg.tetrads]
    return [[float(v) * 100 for v in f] for f in fields]

def generate_comparison_box_graph(model_1: str, name_1: str, model_2: str, name_2: str):
    model_1_path = os.path.join(MODEL_DIR, model_1)
    model_2_path = os.path.join(MODEL_DIR, model_2)

    if not os.path.exists(model_1_path):
        print(f"Model {model_1} not found!")
        return
    
    if not os.path.exists(model_2_path):
        print(f"Model {model_2} not found!")
        return
    
    # Load aggregation files
    normal_1 = os.path.join(model_1_path, "aggregated_data.json")
    aggregated_1 = load_aggregated(normal_1)
    normal_2 = os.path.join(model_2_path, "aggregated_data.json")
    aggregated_2 = load_aggregated(normal_2)

    # Prepare data 
    x = np.arange(len(MIR_LABELS))
    data_1 = to_boxplot_data(aggregated_1)
    data_2 = to_boxplot_data(aggregated_2)

    # Display
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(
        data_1,
        positions=x - WIDTH / 2,
        widths=WIDTH,
        patch_artist=True,
        boxprops=dict(facecolor="C0", alpha=0.7),
        medianprops=dict(color="C0"),
        label=name_1,
        showfliers=False
    )
    bp2 = ax.boxplot(
        data_2,
        positions=x + WIDTH / 2,
        widths=WIDTH,
        patch_artist=True,
        boxprops=dict(facecolor="C1", alpha=0.7),
        medianprops=dict(color="C1"),
        label=name_2,
        showfliers=False
    )

    # Customize graph
    ax.set_xticks(x)
    ax.set_xticklabels(MIR_LABELS)
    ax.set_ylabel('Přesnost (%)')
    ax.set_xlabel("Metrika")
    ax.set_ylim(30, 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'{name_1} vs {name_2}')
    ax.legend(handles=[bp1["boxes"][0], bp2["boxes"][0]], labels=[name_1, name_2])

    # Save graph
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model_1}_{model_2}_comparison.png")
    # plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved to {out_path}")

def generate_crf_box_graph(model: str, name: str):
    model_path = os.path.join(MODEL_DIR, model)

    if not os.path.exists(model_path):
        print(f"Model {model} not found!")
        return
    
    # Load aggregation files
    normal = os.path.join(model_path, "aggregated_data.json")
    aggregated = load_aggregated(normal)
    crf = os.path.join(model_path, "crf_aggregated_data.json")
    crf_aggregated = load_aggregated(crf)

    # Prepare data 
    x = np.arange(len(MIR_LABELS))
    data = to_boxplot_data(aggregated)
    crf_data = to_boxplot_data(crf_aggregated)

    # Display
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(
        data,
        positions=x - WIDTH / 2,
        widths=WIDTH,
        patch_artist=True,
        boxprops=dict(facecolor="C0", alpha=0.7),
        medianprops=dict(color="C0"),
        label=name,
        showfliers=False
    )
    bp2 = ax.boxplot(
        crf_data,
        positions=x + WIDTH / 2,
        widths=WIDTH,
        patch_artist=True,
        boxprops=dict(facecolor="C1", alpha=0.7),
        medianprops=dict(color="C1"),
        label=f"{name} + CRF",
        showfliers=False
    )

    # Customize graph
    ax.set_xticks(x)
    ax.set_xticklabels(MIR_LABELS)
    ax.set_ylabel('Přesnost (%)')
    ax.set_xlabel("Metrika")
    ax.set_ylim(30, 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'{name} - porovnání CRF')
    ax.legend(handles=[bp1["boxes"][0], bp2["boxes"][0]], labels=[name, f"{name} + CRF"])

    # Save graph
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model}_crf_comparison.png")
    # plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved to {out_path}")

def generate_time_box_graph(*trios: tuple[str, str, bool]):
    data = []
    labels = []
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
        labels.append(name)
        data.append(aggregated.inference_times)

    x = np.arange(len(labels))
    # Display
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(
        data,
        positions=x,
        widths=WIDTH,
        patch_artist=True,
        # boxprops=dict(facecolor="C0", alpha=0.7),
        # medianprops=dict(color="C0"),
        # label=name,
        showfliers=False
    )


    # Customize graph
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Čas (s)')
    ax.set_xlabel("Model")
    # ax.set_ylim(30, 100)
    # ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'Čas na predikci')

    # Save graph
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"inference_time_comparison.png")
    # plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    # generate_crf_box_graph("log_pcp_complex", "Logistická regrese")

    # generate_crf_box_graph("cnn_144_complex", "CNN")

    # generate_comparison_box_graph("cr1_144_complex", "CR1", "cr2_144_complex", "CR2")

    generate_time_box_graph(("log_pcp_complex", "Logistická regrese", True), ("cnn_144_complex", "CNN", True), ("cr2_144_complex", "CR2", False), ("btc_144_complex", "BTC", False))

