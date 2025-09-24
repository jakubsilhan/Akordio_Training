import numpy as np
from enum import Enum

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJMIN = ["maj", "min"]
MAJMIN7 = ["maj", "min", "maj7", "min7", "7"]
COMPLEX = ["maj", "min", "maj7", "min7", "7", "dim", "aug", "min6", "maj6", "minmaj7", "dim7", "hdim7", "sus2", "sus4"]

class Complexity(Enum):
    MAJMIN = 1
    MAJMIN7 = 2
    COMPLEX = 3

class Chords:
    
    def __init__(self):
        self.pitches = {
            'maj': self.interval_list('(1,3,5)'),
            'min': self.interval_list('(1,b3,5)'),
            'dim': self.interval_list('(1,b3,b5)'),
            'aug': self.interval_list('(1,3,#5)'),
            'maj7': self.interval_list('(1,3,5,7)'),
            'min7': self.interval_list('(1,b3,5,b7)'),
            '7': self.interval_list('(1,3,5,b7)'),
            '6': self.interval_list('(1,6)'),
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'),
            '1': self.interval_list('(1)'),
            'dim7': self.interval_list('(1,b3,b5,bb7)'),
            'hdim7': self.interval_list('(1,b3,b5,b7)'),
            'minmaj7': self.interval_list('(1,b3,5,7)'),
            'maj6': self.interval_list('(1,3,5,6)'),
            'min6': self.interval_list('(1,b3,5,6)'),
            '9': self.interval_list('(1,3,5,b7,9)'),
            'maj9': self.interval_list('(1,3,5,7,9)'),
            'min9': self.interval_list('(1,b3,5,b7,9)'),
            'sus2': self.interval_list('(1,2,5)'),
            'sus4': self.interval_list('(1,4,5)'),
            '11': self.interval_list('(1,3,5,b7,9,11)'),
            'min11': self.interval_list('(1,b3,5,b7,9,11)'),
            '13': self.interval_list('(1,3,5,b7,13)'),
            'maj13': self.interval_list('(1,3,5,7,13)'),
            'min13': self.interval_list('(1,b3,5,b7,13)')
        }

        self.majmin_encodings = self._generate_encodings(PITCH_CLASS, MAJMIN)
        self.majmin7_encodings = self._generate_encodings(PITCH_CLASS, MAJMIN7)
        self.complex_encodings = self._generate_encodings(PITCH_CLASS, COMPLEX)
        self.complex_encodings.append('X')

    # Encoding methods
    def encode(self, chord: str, type: Complexity) -> int:
        '''
        Encodes a chord string into a numeric value

        Parameters
        ----------
        chord : str
            A string representing a chord (e.g. G:maj)
        type : Complexity
            An enum representing which type of chord encodings is expected

        Returns
        -------
        chord_number : int
            An encoded chord into a numeric value
        '''
        try:
            match type:
                case Complexity.COMPLEX:
                    return self.complex_encodings.index(chord)
                case Complexity.MAJMIN7:
                    return self.majmin7_encodings.index(chord)
                case default:
                    return self.majmin_encodings.index(chord)
        except ValueError:
            return 0
        
    def decode(self, number: int, type: Complexity) -> str:
        '''
        Decodes a chord string from a numeric value

        Parameters
        ----------
        number : int
            A number representing an encoded chord
        type : Complexity
            An enum representing which type of chord encodings is expected

        Returns
        -------
        chord : str
            A string of the decoded chord
        '''
        try:
            match type:
                case Complexity.COMPLEX:
                    return self.complex_encodings[number]
                case Complexity.MAJMIN7:
                    return self.majmin7_encodings[number]
                case default:
                    return self.majmin_encodings[number]
        except IndexError:
            return "N"

    def _generate_encodings(self, pitch_classes: list, qualities: list) -> list:
        '''
        Reduces encoding list for pitch classes and qualities

        Parameters
        ----------
        pitch_classes : list
            List of pitch qualities to base encodings on
        qualities : list
            List of qualities to base encodings on

        Returns
        -------
        chords : list
            List of all possible pitch-quality combinations
        '''
        chords = []
        chords.append("N")
        for pitch in pitch_classes:
            for quality in qualities:
                chords.append(f"{pitch}:{quality}")
        return chords

    # Reduction methods
    def reduce(self, chord: str, complexity: Complexity) -> str:
        match complexity:
            case Complexity.COMPLEX:
                return self.complex(chord)
            case Complexity.MAJMIN7:
                return self.majmin7(chord)
            case default:
                return self.majmin(chord)

    def majmin(self, chord: str) -> str:
        '''
        Reduces a chord label to its root and basic quality: maj or min.

        Parameters
        ----------
        chord : str
            e.g. "C:min7", "D:maj13"

        Returns
        -------
        reduced_chord : str
            e.g. "C:min" or "D:maj"
        '''
        root, pitch_classes = self.deconstruct_chord(chord)

        if root=="N":
            return "N"
        if pitch_classes[4]:  # major third
            return f"{root}:maj"
        elif pitch_classes[3]:  # minor third
            return f"{root}:min"
        else:
            return f"{root}:maj"  # default to maj
    
    def majmin7(self, chord: str) -> str:
        '''
        Reduces a chord label to maj, min, 7, maj7, or min7 based on its interval content.

        Parameters
        ----------
        chord : str
            e.g. "C:maj13", "D:min9", "G:7"

        Returns
        -------
        reduced_chord : str
            e.g. "C:maj7", "D:min7", "G:7"
        '''
        root, pitch_classes = self.deconstruct_chord(chord)

        if root == "N":
            return "N"

        has_major_3rd = pitch_classes[4]    # major third
        has_minor_3rd = pitch_classes[3]    # minor third
        has_minor_7th = pitch_classes[10]   # minor seventh
        has_major_7th = pitch_classes[11]   # major seventh

        if has_major_3rd:
            if has_minor_7th:
                return f"{root}:7"
            elif has_major_7th:
                return f"{root}:maj7"
            else:
                return f"{root}:maj"

        elif has_minor_3rd:
            if has_minor_7th:
                return f"{root}:min7"
            else:
                return f"{root}:min"

        else:
            return f"{root}:maj"  # Default fallback
    
    def complex(self, chord: str) -> str:
        '''
        Reduces a chord label to one of the canonical COMPLEX chord types
        based on its interval content.

        Parameters
        ----------
        chord : str
            e.g. "C:maj13", "D:min9", "G:dim7"

        Returns
        -------
        reduced_chord : str
            e.g. "C:maj7", "D:min7", "G:dim7"
        '''
        root, pc = self.deconstruct_chord(chord)

        if root == "N":
            return "N"

        # Define pitch class intervals for checking
        has = lambda i: pc[i % 12]

        m3 = has(3)
        M3 = has(4)
        d5 = has(6)
        P5 = has(7)
        A5 = has(8)
        m6 = has(8)
        M6 = has(9)
        m7 = has(10)
        M7 = has(11)
        sus2 = has(2) and not m3 and not M3
        sus4 = has(5) and not m3 and not M3

        if sus2:
            return f"{root}:sus2"
        elif sus4:
            return f"{root}:sus4"
        elif m3 and d5 and has(9):
            return f"{root}:dim7"
        elif m3 and d5 and m7:
            return f"{root}:hdim7"
        elif m3 and d5:
            return f"{root}:dim"
        elif M3 and A5:
            return f"{root}:aug"
        elif m3 and m7 and M6:
            return f"{root}:min6"
        elif M3 and M6 and not m7:
            return f"{root}:maj6"
        elif m3 and M7:
            return f"{root}:minmaj7"
        elif M3 and M7:
            return f"{root}:maj7"
        elif m3 and m7:
            return f"{root}:min7"
        elif M3 and m7:
            return f"{root}:7"
        elif M3:
            return f"{root}:maj"
        elif m3:
            return f"{root}:min"
        else:
            return f"X"  # Uknown

    # Interval methods
    def deconstruct_chord(self, chord: str) -> tuple:
        '''
        Splits a chord into its root and intervals

        Parameters
        ----------
        chord : str
            Chord label as a string

        Returns
        -------
        deconstructed chord : tuple
            A tuple containing the root and binary pitch class vector
        '''
        if chord=="N":
            return ("N", None)
        root, quality = chord.split(":")[:2]
        quality_pitches = self.pitches.get(quality)
        if quality_pitches is None:
            # print(f"Quality not recognized: {chord}")
            quality_pitches = self.pitches.get("maj")
        return (root, quality_pitches)


    def interval_list(self, intervals_str: str):
        """
        Convert a list of intervals given as string to a binary pitch class
        representation. For example, '1, 3, 5' would become
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

        Parameters
        ----------
        intervals_str : str
            List of intervals as comma-separated string (e.g. '1, 3, 5').

        Returns
        -------
        pitches : numpy array
            Binary pitch class representation of chord intervals

        """

        # Clean and split the input string
        interval_str = intervals_str.strip('() ')
        intervals = [i.strip() for i in interval_str.split(',') if i.strip()]


        # Map intervals to semitone offsets
        interval_map = {
            '1': 0,
            'b2': 1, '♭2': 1,
            '2': 2,
            '#2': 3, 'b3': 3,
            '3': 4,
            '4': 5,
            '#4': 6, 'b5': 6,
            '5': 7,
            '#5': 8, 'b6': 8,
            '6': 9,
            'bb7': 9,
            'b7': 10,
            '7': 11,
            'b9': 13,
            '9': 14,
            '#9': 15,
            '11': 17,
            '#11': 18,
            '13': 21
        }

        # Initialize a 12-element one-hot vector
        pitches = np.zeros(12, dtype=int)

        for i in intervals:
            semitone = interval_map.get(i)
            if semitone is None:
                raise ValueError(f"Unrecognized interval: {i}")
            pitches[semitone % 12] = 1
        return pitches

if __name__ == "__main__":
    chord_tool = Chords()
    chord = "C:maj7"
    complexity = Complexity.MAJMIN
    print(chord_tool.deconstruct_chord(chord))
    print(chord_tool.reduce(chord, complexity))