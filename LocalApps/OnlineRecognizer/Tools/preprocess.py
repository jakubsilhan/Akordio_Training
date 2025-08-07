import os, librosa, torch
import numpy as np
from Core.config import Config

def process_audio(y, config: Config) -> torch.Tensor:
    # Extract features
    x = extract_features(y, config)

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    return x_tensor

def extract_features(x, config: Config) -> np.ndarray:
    if config.data.preprocess.pcp.enabled:
        features = librosa.feature.chroma_cqt(y=x, sr=config.data.preprocess.sampling_rate, bins_per_octave=config.data.preprocess.bins_per_octave, hop_length=config.data.preprocess.hop_length, n_chroma=config.data.preprocess.pcp.bins, n_octaves=config.data.preprocess.pcp.octaves)
    else:
        features = np.abs(librosa.cqt(x, sr=config.data.preprocess.sampling_rate, bins_per_octave=config.data.preprocess.bins_per_octave,n_bins=config.data.preprocess.cqt_bins, hop_length=config.data.preprocess.hop_length))
        features = librosa.amplitude_to_db(features, ref=np.max)

    features = features.T

    return features