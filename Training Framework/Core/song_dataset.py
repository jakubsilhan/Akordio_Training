from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from core.net_config import Config

class SongDataset(Dataset):
    def __init__(self, song_tensors, config: Config):
        """
        Initializes the dataset with song samples.
        Each song sample is a tuple pf tensors (X, y), where:
        X = sequence of feature vectors (chroma/cqt)
        y = sequence of labels (maj_min)
        """
        self.config = config
        self.song_tensors = song_tensors

    def __len__(self):
        return len(self.song_tensors)

    def __getitem__(self, idx):
        X, y = self.song_tensors[idx]
        # TODO Check
        if not self.config.data.preprocess.pcp.enabled:
            X = torch.log(X+1e-6)
        X = X.to(torch.float32)
        return X, y

# Ensures padding
def make_collate_fn(padding_index):
    def collate_fn(batch):
        # Sort the batch by the length of X (descending order)
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Extract X (features) and y (labels) from batch
        X_batch, y_batch = zip(*batch)

        # Pad sequences in X (features)
        X_batch_padded = pad_sequence(X_batch, batch_first=True, padding_value=0) # type: ignore
        # For labels, pad them to the same length as the longest sequence
        y_batch_padded = pad_sequence(y_batch, batch_first=True, padding_value=padding_index)  # type: ignore Use -1 for padding labels
        
        return X_batch_padded, y_batch_padded
    return collate_fn