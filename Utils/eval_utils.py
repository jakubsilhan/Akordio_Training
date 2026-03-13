import json
from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class AggregatedData():
    epoch_counts: List[int] = field(default_factory=list)
    train_times: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    root: List[float] = field(default_factory=list)
    thirds: List[float] = field(default_factory=list)
    sevenths: List[float] = field(default_factory=list)
    triads: List[float] = field(default_factory=list)
    tetrads: List[float] = field(default_factory=list)
    segmentation: List[float] = field(default_factory=list)
    majmin: List[float] = field(default_factory=list)

def save_aggregated(filepath: str, data: AggregatedData) -> None:
    with open(filepath, "w") as f:
        json.dump(asdict(data), f, indent=3)

def load_aggregated(filepath: str) -> AggregatedData:
    with open(filepath, "r") as f:
        json_data = json.load(f)
        data = AggregatedData(**json_data)
    return data
