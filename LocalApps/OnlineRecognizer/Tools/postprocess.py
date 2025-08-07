import torch, os
from Core.config import Config

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJMIN = ["maj", "min"]
MAJMIN7 = ["maj", "min", "maj7", "min7", "7"]
COMPLEX = ["maj", "min", "maj7", "min7", "7", "dim", "aug", "min6", "maj6", "minmaj7", "dim7", "hdim7", "sus2", "sus4"]


def decode(pred, config: Config) -> str:
    match config.train.model_complexity:
        case "complex":
            encodings = _generate_encodings(PITCH_CLASS, COMPLEX)
        case "majmin7":
            encodings = _generate_encodings(PITCH_CLASS, MAJMIN7)
        case default:
            encodings = _generate_encodings(PITCH_CLASS, MAJMIN)
    
    try:
        chord = encodings[pred]
    except IndexError:
        chord = ("N")

    return chord

def _generate_encodings(pitch_classes: list, qualities: list) -> list:
    chords = []
    chords.append("N")
    for pitch in pitch_classes:
        for quality in qualities:
            chords.append(f"{pitch}:{quality}")
    return chords