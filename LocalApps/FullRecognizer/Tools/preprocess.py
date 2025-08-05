import os, librosa, torch
import numpy as np
from Core.config import Config

def process_audio(audio_path: str, config: Config) -> tuple[torch.Tensor, list[float]]:
    if not os.path.exists(audio_path):
        raise ValueError("Nonexistent audio file input!")
    
    # Load audio
    x, sr = librosa.load(audio_path, sr=config.data.preprocess.sampling_rate)

    # Extract features
    x, timestamps = extract_features(x, config)

    # TODO consider splitting into fragments here

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    return x_tensor, timestamps

def extract_features(x, config: Config) -> tuple[np.ndarray, list[float]]:
    if config.data.preprocess.pcp.enabled:
        features = librosa.feature.chroma_cqt(y=x, sr=config.data.preprocess.sampling_rate, bins_per_octave=config.data.preprocess.bins_per_octave, hop_length=config.data.preprocess.hop_length, n_chroma=config.data.preprocess.pcp.bins, n_octaves=config.data.preprocess.pcp.octaves)
    else:
        features = np.abs(librosa.cqt(x, sr=config.data.preprocess.sampling_rate, bins_per_octave=config.data.preprocess.bins_per_octave,n_bins=config.data.preprocess.cqt_bins, hop_length=config.data.preprocess.hop_length))
        features = librosa.amplitude_to_db(features, ref=np.max)

    features = features.T
    times = librosa.frames_to_time(np.arange(features.shape[0]), sr=config.data.preprocess.sampling_rate, hop_length=config.data.preprocess.hop_length)

    return features, times