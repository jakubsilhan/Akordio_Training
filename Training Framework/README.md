# chord-recognition
A repo for chord recognition

## Configuration
This is what the configuration file, that controls the whole training process, looks like.
```
base:
  random_seed: 42
data:
  dataset_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen]
  preprocessed_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/FullSongDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (duration of a frame if equal to window_size)
    fragment_size: 0 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 6 # highest pitch shift
    sampling_rate: 22050 # sampling rate
train: # used for both training a evaluating
  data_source: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/BaseDataset"
  model_path: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Models"
  model_name: "cr1_144_majmin7"
  test_fold: 2 # which fold to use as test set
  model_type: "cr1"
  model_complexity: "majmin7" # which type of chords to train (majmin/majmin7/complex)
  checkpoint_interval: 50
  model:
    batch_size: 128
    input: 144 # feature size
    output: 61 # output size (number of chords 25/61/170)
    hidden: [128] # hidden size (depends on a model)
    dropout: [0.4, 0, 0] # dropout (depends on a model)
    layers: 2 # number of layers
    bidirectional: True
    padding_index: -1
    epoch_count: 100
    learning_rate: 0.0001
    weight_decay: 1e-5
    scheduler_step: 100
    scheduler_gamma: 0.1
    normalization: True
```

## Models
### CR1
 - A CRNN encoder model based on CR1 model from [STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD
RECOGNITION](https://brianmcfee.net/papers/ismir2017_chord.pdf)

#### Model specific configurations
- dropout = 3 value array
- hidden = 1 value array

### SimpleLSTM
 - A simple lstm model mainly used in conjunction with extracted chroma features

#### Model specific configurations
- dropout = 1 value array
- hidden = 1 value array

## Datasets
### BaseDataset
```
data:
  dataset_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen]
  preprocessed_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/BaseDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (duration of a frame if equal to window_size)
    fragment_size: 108 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 6 # highest pitch shift
    sampling_rate: 22050 # sampling rate
```

### FullSongDataset
```
data:
  dataset_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen]
  preprocessed_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/FullSongDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (duration of a frame if equal to window_size)
    fragment_size: 0 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 6 # highest pitch shift
    sampling_rate: 22050 # sampling rate
```


# Temporary
## This worked for online
```
base:
  random_seed: 42
data:
  dataset_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen]
  preprocessed_dir: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/FullSongDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (duration of a frame if equal to window_size)
    fragment_size: 0 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 6 # highest pitch shift
    sampling_rate: 22050 # sampling rate
train: # used for both training a evaluating
  data_source: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Data/BaseDataset"
  model_path: "D:/Development/Source/University/Diploma/chord-recognition/Training Framework/Models"
  model_name: "cr1_144_majmin_online"
  test_fold: 2 # which fold to use as test set
  model_type: "cr1"
  model_complexity: "majmin" # which type of chords to train (majmin/majmin7/complex)
  checkpoint_interval: 50
  model:
    batch_size: 128
    input: 144 # feature size
    output: 25 # output size (number of chords 25/61/170)
    hidden: [128] # hidden size (depends on a model)
    dropout: [0.4, 0, 0] # dropout (depends on a model)
    layers: 2 # number of layers
    bidirectional: False
    padding_index: -1
    epoch_count: 100
    learning_rate: 0.0001
    weight_decay: 1e-5
    scheduler_step: 100
    scheduler_gamma: 0.1
    normalization: False # Not working for now
```