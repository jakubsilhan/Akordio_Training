import os
import Tools.preprocess as preprocess
import Tools.predict as predict
import Tools.postprocess as postprocess
from tkinter import Tk
from tkinter.filedialog import  askopenfilename
from Core.config import Config, load_config

def Run():
    # Loading config
    config = load_config("Model/config.yaml")

    # File dialog
    Tk().withdraw()
    filename = askopenfilename()
    song_name = os.path.basename(filename).split(".")[0]

    # Process audio
    x_tensor, timestamps = preprocess.process_audio(filename, config)

    # Run classification
    preds = predict.classify_chords(x_tensor, config)

    # Save annotation
    postprocess.create_annotation(preds, config, song_name)


if __name__ == "__main__":
    Run()