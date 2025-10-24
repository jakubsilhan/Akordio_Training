import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import pyrubberband as pyrb
import os, librosa, shutil, io, torch
from sklearn.model_selection import KFold
from .net_config import Config, load_config

class Preprocessor():
    def __init__(self, config: Config):
        self.config = config

    def process_all_data(self) -> None:
        '''
        Goes through all specified data and processes them into a dataset according to configuration
        '''
        if os.path.exists(self.config.data.preprocessed_dir):
            shutil.rmtree(self.config.data.preprocessed_dir)
        os.makedirs(self.config.data.preprocessed_dir)
        for dataset in tqdm(self.config.data.datasets, desc="Processing datasets"):
            self.process_dataset(dataset)
        shutil.copy2("config.yaml", self.config.data.preprocessed_dir)

    def process_dataset(self, dataset: str) -> None:
        '''
        Goes through all songs in a dataset a processes them
        '''
        dataset_path = os.path.join(self.config.data.dataset_dir, dataset)
        all_files = []
        for path, _, files in os.walk(os.path.join(dataset_path, "Audio")):
            for filename in files:
                all_files.append((path, filename))

        kf = KFold(self.config.data.preprocess.num_splits, shuffle=True, random_state=self.config.base.random_seed)
        fold_pbar = tqdm(total=self.config.data.preprocess.num_splits, desc="Generating folds")
        for fold, (_, fold_indices) in enumerate(kf.split(all_files)):
            fold_path = os.path.join(self.config.data.preprocessed_dir, str(fold))
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            fold_files = [all_files[idx] for idx in fold_indices]
            for path, filename in tqdm(fold_files, desc="Processing fold songs"):
                self.process_song(path, filename, fold)
            fold_pbar.update(1)
        fold_pbar.close()

    def process_song(self, path: str, filename: str, fold: int) -> None:
        '''
        Processes a song and applies pitch shifting aswell if needed
        '''
        audio_path = os.path.join(path, filename)
        y, sr = librosa.load(audio_path, sr=self.config.data.preprocess.sampling_rate)
        intervals = self.load_annotation(path, filename)
        shifts = range(self.config.data.preprocess.pitch_shift_start, self.config.data.preprocess.pitch_shift_end+1)
        for shift_factor in shifts:
            # y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_factor, bins_per_octave=12) # semitone shift -> 12 bins per octave
            if shift_factor != 0:
                y_shifted = pyrb.pitch_shift(y, sr=sr, n_steps=shift_factor)
            else:
                y_shifted = y
            features, times = self.process_features(y_shifted)
            intervals_shifted = self.shift_annotation(intervals, shift_factor)
            labels = self.assign_labels_to_times(times, intervals_shifted)

            song_df = pd.concat([
              pd.DataFrame({"timestamp": times}),
              pd.DataFrame(features),
              pd.DataFrame({"chord": labels})
            ], axis=1)

            # Saving
            save_base = filename.replace(".mp3", "").split("_-_")[-1]
            self.save_fragments(song_df, save_base, fold, shift_factor)

    def process_audio(self, audio: bytes) -> list[torch.Tensor]:
        """
        Processes audio into features according to the config
        """
        
        # Load audio
        audio_buffer = io.BytesIO(audio)
        audio_buffer.seek(0)
        x, sr = librosa.load(audio_buffer, sr=self.config.data.preprocess.sampling_rate)

        # Extract features
        features,_ = self.process_features(x)

        # Split into fragments
        fragment_size = self.config.data.preprocess.fragment_size
        fragments = []

        # Return with no fragmenting
        if fragment_size == 0:
            fragments.append(torch.tensor(features, dtype=torch.float64))
            return fragments
        
        # Fragment
        hop_size = int(self.config.data.preprocess.fragment_size * self.config.data.preprocess.fragment_hop)
        for start in range(0, len(features), hop_size):
            fragment = features[start:start+fragment_size]
            if len(fragment) < fragment_size:
                pad_width = fragment_size - len(fragment)
                fragment = np.pad(fragment, ((0, pad_width), (0, 0)), mode='constant')
            fragments.append(torch.tensor(fragment, dtype=torch.float64))
        
        return fragments

    def process_features(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Processes audio into a log CQT or chromagram
        '''
        if self.config.data.preprocess.pcp.enabled:
            features = librosa.feature.chroma_cqt(y=y, sr=self.config.data.preprocess.sampling_rate, bins_per_octave=self.config.data.preprocess.bins_per_octave, hop_length=self.config.data.preprocess.hop_length, n_chroma=self.config.data.preprocess.pcp.bins, n_octaves=self.config.data.preprocess.pcp.octaves)
        else:
            cqt = np.abs(librosa.cqt(y, sr=self.config.data.preprocess.sampling_rate, bins_per_octave=self.config.data.preprocess.bins_per_octave,n_bins=self.config.data.preprocess.cqt_bins, hop_length=self.config.data.preprocess.hop_length))
            # features = librosa.amplitude_to_db(cqt, ref=np.max)
            # features = librosa.power_to_db(cqt**2, ref=np.max)
            # features = np.log(cqt + 1e-6)
            features = cqt
    
        features = features.T
        times = librosa.frames_to_time(np.arange(features.shape[0]), sr=self.config.data.preprocess.sampling_rate, hop_length=self.config.data.preprocess.hop_length)

        return features, times

    def load_annotation(self, path: str, filename: str) -> list[tuple[float, float, str]]:
        '''
        Loads annotation into a list of tuples
        '''
        path = path.replace("Audio", "Chords")
        filename = filename.replace("mp3", "lab")
        filepath = os.path.join(path, filename)
        intervals = []
        with open(filepath) as f:
            for line in f:
                start, end, chord = line.strip().split()
                intervals.append((float(start), float(end), chord))

        return intervals
    
    def shift_annotation(self, intervals: list[tuple[float, float, str]], shift_factor) -> list:
        '''
        Shift labels in the annotations according to the shift factor
        '''
        shifted = []
        for start, end, label in intervals:
            new_label = self.shift_root(label, shift_factor)
            shifted.append((start, end, new_label))
        return shifted

    def shift_root(self, chord: str, semitone_shift: int) -> str:
        '''
        Shifts the label root according to the shift factor
        '''
        # TODO consider keeping the bass and adds (not used for now)
        root, type_name, adds, bass = chord, '', [], ''

        note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if 'N' in chord:
            return 'N'

        if '/' in chord:
            chord, bass = chord.split("/")
            root = chord
        
        if '(' in chord:
            adds = chord[chord.find('(') + 1:chord.find(')')].split(',')
            chord = chord[:chord.find('(')] + chord[chord.find(')') + 1:]
            root = chord

        if ':' in chord:
            root, type_name = chord.split(':', maxsplit=1)
            if type_name == '':
                type_name = 'maj'
        elif len(chord) >= 2 and chord[1] in ['b', '#']:
            root = chord[:2]
            type_name = chord[2:]
        else:
            root = chord[:1]
            type_name = 'maj'

        root_note = self.normalize_note(root)
        idx = note_list.index(root_note)
        new_idx = (idx + semitone_shift) % 12

        return note_list[new_idx]+":"+type_name

    # Utils
    def save_fragments(self, song_df: pd.DataFrame, base_name: str, fold: int, shift_factor: int) -> None:
        '''
        Splits the song dataframe into fixed-size fragments (in frames) and saves them as individual npz files
        '''
        if self.config.data.preprocess.pcp.enabled:
            input_dim = self.config.data.preprocess.pcp.bins
        else:
            input_dim = self.config.data.preprocess.cqt_bins

        # Full song mode
        if self.config.data.preprocess.fragment_size <= 0:
            # Extract into numpy arrays
            timestamps = song_df.iloc[:, 0].values.astype(np.float64)
            X = song_df.iloc[:, 1:1 + input_dim].values.astype(np.float64) # skip timestamp
            y = song_df["chord"].values.astype(str)

            # Prepare pathing
            song_filename = f"{base_name}_shift{shift_factor:02d}.npz"
            song_path = os.path.join(self.config.data.preprocessed_dir, str(fold), song_filename)
            os.makedirs(os.path.dirname(song_path), exist_ok=True)

            # Save into npz
            np.savez_compressed(song_path, timestamps=timestamps, X=X, y=y)
            return

        # Fragmenting mode
        num_rows = len(song_df)
        hop_size = int(self.config.data.preprocess.fragment_size * self.config.data.preprocess.fragment_hop)
        for start in range(0, num_rows, hop_size):
            fragment = song_df.iloc[start:start + self.config.data.preprocess.fragment_size]

            # If fragment is shorter than desired, pad with zeros for features, "N" for chords
            # if len(fragment) < self.config.data.preprocess.fragment_size:
            #     pad_len = self.config.data.preprocess.fragment_size - len(fragment)
            #     pad_features = np.zeros((pad_len, input_dim), dtype=np.float32)
            #     pad_chords = np.array(["N"] * pad_len)
            #     fragment_features = fragment.iloc[:, 1:1 + input_dim].values.astype(np.float32)
            #     fragment_chords = fragment["chord"].values.astype(str)
            #     X = np.vstack([fragment_features, pad_features])
            #     y = np.hstack([fragment_chords, pad_chords])
            #     timestamps = np.concatenate([fragment.iloc[:, 0].values.astype(np.float32), np.zeros(pad_len, dtype=np.float32)])
            # else:
            if len(fragment) < self.config.data.preprocess.fragment_size:
                continue

            X = fragment.iloc[:, 1:1 + input_dim].values.astype(np.float64)
            y = fragment["chord"].values.astype(str)
            timestamps = fragment.iloc[:, 0].values.astype(np.float64)

            # Prepare path
            frag_filename = f"{base_name}_shift{shift_factor:02d}_frag{start//hop_size:04d}.npz"
            frag_path = os.path.join(self.config.data.preprocessed_dir, str(fold), frag_filename)
            os.makedirs(os.path.dirname(frag_path), exist_ok=True)

            # Save fragment
            np.savez_compressed(frag_path, timestamps=timestamps, X=X, y=y)



    def assign_labels_to_times(self, times, intervals) -> np.ndarray:
        '''
        Creates an array of chord alligned to specific timings
        '''
        labels = []
        for t in times:
            found = False
            for start, end, chord in intervals:
                if start <= t and t < end:
                    labels.append(chord)
                    found = True
                    break
            if not found:
                labels.append("N")
        
        return np.array(labels)
    
    # def assign_labels_to_times(self, times: np.ndarray, intervals: list[tuple[float, float, str]]) -> np.ndarray:
    #     '''
    #     Creates an array of chords aligned to CQT frames.
    #     '''
    #     hop_length = self.config.data.preprocess.hop_length
    #     sr = self.config.data.preprocess.sampling_rate
    #     time_interval = hop_length / sr
        
    #     labels = []
        
    #     for t in times:
    #         frame_end = t + time_interval
    #         best_overlap = 0.0
    #         best_label = "N"
            
    #         # Find chord with maximum overlap
    #         for start, end, chord in intervals:
    #             overlap = max(0.0, min(end, frame_end) - max(start, t))
    #             if overlap > best_overlap:
    #                 best_overlap = overlap
    #                 best_label = chord
            
    #         labels.append(best_label)
    
    #     return np.array(labels)


    def normalize_note(self, note: str) -> str:
        '''
        Normalizes flats
        '''
        flat_to_sharp = {
            'Cb': 'B',
            'Db': 'C#',
            'Eb': 'D#',
            'Fb': 'E',
            'Gb': 'F#',
            'Ab': 'G#',
            'Bb': 'A#',
            'E#': 'F',
            'B#': 'C'
        }
        if note in flat_to_sharp:
            return flat_to_sharp[note]
        return note
    

# if __name__ == "__main__":
#     config = load_config("config.yaml")
#     preprocessing = Preprocess(config)
#     preprocessing.process_all_data()
