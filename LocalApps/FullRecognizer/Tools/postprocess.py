import torch, os
from Core.config import Config

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJMIN = ["maj", "min"]
MAJMIN7 = ["maj", "min", "maj7", "min7", "7"]
COMPLEX = ["maj", "min", "maj7", "min7", "7", "dim", "aug", "min6", "maj6", "minmaj7", "dim7", "hdim7", "sus2", "sus4"]

def create_annotation(preds: torch.Tensor, config: Config, song_name: str):
    # Decode preds
    chords = decode(preds, config)

    # Generate annotation
    labels, intervals = generate_intervals(chords, config)

    # Generate annotation string
    data = get_interval_string(labels, intervals)

    # Write annotation to file
    output_path = os.path.join("Output", song_name+".lab")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        file.write(data)

def get_interval_string(labels: list[str], intervals: list[tuple[float, float]]):
    lines = []
    for label, (start, end) in zip(labels, intervals):
        lines.append(f"{start:.3f} {end:.3f} {label}")
    return "\n".join(lines)


def generate_intervals(chords: list, config: Config) -> tuple[list[str], list[tuple]]:
    start = 0
    frame_duration = config.data.preprocess.hop_length/config.data.preprocess.sampling_rate
    intervals = list()
    labels = list()
    for i in range(len(chords)):
        if i == 0:
            prev_chord = chords[i]
            continue

        if chords[i] != prev_chord:
            start_time = start * frame_duration
            end_time = i * frame_duration
            intervals.append([start_time, end_time])
            labels.append(prev_chord)
            prev_chord = chords[i]
            start = i
        
    return labels, intervals


def decode(preds: torch.Tensor, config: Config) -> list[str]:
    chords = list()
    match config.train.model_complexity:
        case "complex":
            encodings = _generate_encodings(PITCH_CLASS, COMPLEX)
        case "majmin7":
            encodings = _generate_encodings(PITCH_CLASS, MAJMIN7)
        case default:
            encodings = _generate_encodings(PITCH_CLASS, MAJMIN)
    
    for pred in preds:
        try:
            chords.append(encodings[pred])
        except IndexError:
            chords.append("N")

    return chords

def _generate_encodings(pitch_classes: list, qualities: list) -> list:
    chords = []
    chords.append("N")
    for pitch in pitch_classes:
        for quality in qualities:
            chords.append(f"{pitch}:{quality}")
    return chords