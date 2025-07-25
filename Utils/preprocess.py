import numpy as np
import pandas as pd
from tqdm import tqdm
import os, librosa, shutil
from sklearn.model_selection import KFold
from Core.config import Config, load_config

class Preprocess():
    def __init__(self, config: Config):
        self.config = config

    def process_all_data(self):
        '''
        Goes through all specified data and processes them into a dataset according to configuration
        '''
        if os.path.exists(self.config.data.preprocessed_dir):
            shutil.rmtree(self.config.data.preprocessed_dir)
        os.makedirs(self.config.data.preprocessed_dir)
        for dataset in tqdm(config.data.datasets, desc="Processing datasets"):
            self.process_dataset(dataset)

    def process_dataset(self, dataset):
        '''
        Goes through all songs in a dataset a processes them
        '''
        dataset_path = os.path.join(config.data.dataset_dir, dataset)
        all_files = []
        for path, _, files in os.walk(os.path.join(dataset_path, "Audio")):
            for filename in files:
                all_files.append((path, filename)) # dataset/Audio/path, song.mp3

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

    def process_song(self, path: str, filename: str, fold: int):
        '''
        Processes a song and applies pitch shifting aswell if needed
        '''
        audio_path = os.path.join(path, filename)
        y, sr = librosa.load(audio_path, sr=self.config.data.preprocess.sampling_rate)
        intervals = self.load_annotation(path, filename)
        shifts = range(self.config.data.preprocess.pitch_shift_start, self.config.data.preprocess.pitch_shift_end+1)
        for shift_factor in shifts:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_factor, bins_per_octave=12) # semitone shift -> 12 bins per octave
            features, times = self.process_features(y_shifted)
            intervals_shifted = self.shift_annotation(intervals, shift_factor)
            labels = self.assign_labels_to_times(times, intervals_shifted)

            song_df = pd.concat([
              pd.DataFrame({"timestamp": times}),
              pd.DataFrame(features),
              pd.DataFrame({"chord": labels})
            ], axis=1)

            # Maybe other edits

            # Saving
            save_base = filename.replace(".mp3", "").split("_-_")[-1]
            self.save_fragments(song_df, save_base, fold, shift_factor)

    def process_features(self, y):
        '''
        Processes audio into a log CQT
        '''
        C = np.abs(librosa.cqt(y, sr=self.config.data.preprocess.sampling_rate, bins_per_octave=self.config.data.preprocess.bins_per_octave,n_bins=self.config.data.preprocess.cqt_bins, hop_length=self.config.data.preprocess.hop_length))
        C = librosa.amplitude_to_db(C, ref=np.max)
        C = C.T
        times = librosa.frames_to_time(np.arange(C.shape[0]), sr=self.config.data.preprocess.sampling_rate, hop_length=self.config.data.preprocess.hop_length)

        return C, times

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
    
    def shift_annotation(self, intervals: list[tuple[float, float, str]], shift_factor):
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
        # TODO consider keeping the bass and adds
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
        else:
            root = chord
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
        num_rows = len(song_df)
        num_fragments = num_rows // self.config.data.preprocess.fragment_size
        input_dim = config.train.model.input

        for i in range(num_fragments):
            fragment = song_df.iloc[i * self.config.data.preprocess.fragment_size : (i + 1) * self.config.data.preprocess.fragment_size]

            # Extract into numpy arrays
            timestamps = fragment.iloc[:, 0].values.astype(np.float32)
            X = fragment.iloc[:, 1:1 + input_dim].values.astype(np.float32)
            y = fragment["chord"].values.astype(str)

            # Prepare pathing
            frag_filename = f"{base_name}_shift{shift_factor:02d}_frag{i:03d}.npz"
            frag_path = os.path.join(self.config.data.preprocessed_dir, str(fold),frag_filename)
            os.makedirs(os.path.dirname(frag_path), exist_ok=True)

            # Save into npz
            np.savez_compressed(frag_path, timestamps=timestamps, X=X, y=y)



    def assign_labels_to_times(self, times, intervals):
        '''
        Creates an array of chord alligned to specific timings
        '''
        labels = []
        for t in times:
            found = False
            for start, end, chord in intervals:
                if start < t and t < end:
                    labels.append(chord)
                    found = True
                    break
            if not found:
                labels.append("N")
        
        return np.array(labels)
    
    def normalize_note(self, note):
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
    

if __name__ == "__main__":
    config = load_config("config.yaml")
    preprocessing = Preprocess(config)
    preprocessing.process_all_data()
