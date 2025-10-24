from pydantic import BaseModel
import yaml

class PCPConfig(BaseModel):
    enabled: bool
    bins: int
    octaves: int

class PreprocessingConfig(BaseModel):
    pcp: PCPConfig
    num_splits: int
    bins_per_octave: int
    cqt_bins: int
    hop_length: int
    fragment_size: int
    fragment_hop: float
    pitch_shift_start: int
    pitch_shift_end: int
    sampling_rate: int

class DataConfig(BaseModel):
    dataset_dir: str
    datasets: list
    preprocessed_dir: str
    preprocess: PreprocessingConfig

class ModelConfig(BaseModel):
    batch_size: int
    input: int
    output: int
    hidden: list
    dropout: list
    layers: int
    bidirectional: bool
    padding_index: int
    loss_patience: int
    epoch_count: int
    learning_rate: float
    weight_decay: float
    scheduler_step: int
    scheduler_gamma: float
    normalization: bool

class TrainConfig(BaseModel):
    data_source: str
    model_path: str
    model_name: str
    test_fold: int
    model_type: str
    model_complexity: str
    checkpoint_interval: int
    model: ModelConfig

class BaseConfig(BaseModel):
    random_seed: int

class Config(BaseModel):
    base: BaseConfig
    data: DataConfig
    train: TrainConfig

def load_config(path="config.yaml") -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)