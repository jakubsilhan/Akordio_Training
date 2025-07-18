from pydantic import BaseModel
import yaml

class PreprocessingConfig(BaseModel):
    pcp: bool
    num_splits: int
    bins_per_octave: int
    cqt_bins: int
    hop_length: int
    fragment_size: int
    pitch_shift_start: int
    pitch_shift_end: int
    sampling_rate: int

class DataConfig(BaseModel):
    dataset_dir: str
    datasets: list
    preprocessed_dir: str
    preprocess: PreprocessingConfig


class ModelConfig(BaseModel):
    bidirectional: bool

class BaseConfig(BaseModel):
    random_seed: int

class Config(BaseModel):
    base: BaseConfig
    data: DataConfig
    model: ModelConfig

def load_config(path="config.yaml") -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)