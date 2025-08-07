import torch, os, pyaudio, math
import numpy as np
import Tools.preprocess as preprocess
import Tools.postprocess as postprocess

from Core.config import Config, load_config
from Model.Model import Model

recording = True

def Run():
    # Initialization
    FORMAT = pyaudio.paInt16 # 16 bit
    CHANNELS = 1 # Mono
    CHUNK = 2048 # Buffer size
    DURATION = 0.5 # Duration to save in seconds

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config = load_config("Model/config.yaml")
    RATE = config.data.preprocess.sampling_rate

    # Load model
    # TODO try bidirectional and bigger hop_length
    model = Model(config=config, device=device).to(device)
    model_path = os.path.join("Model", "final_model.pt")
    loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(loaded["model"])
    model.eval()

    # Start listening
    audio = pyaudio.PyAudio()
    global recording
    stream = audio.open(format=FORMAT, 
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("Background recording started! Press Ctrl+C to stop.")

    try:
        while recording:
            frames = []
            for _ in range(math.ceil(RATE/CHUNK*DURATION)): # Collect n seconds of data
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            # Convert frames to a numpy array
            audio_data = np.concatenate(frames).astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data /= max_val

            # Process audio
            x_tensor = preprocess.process_audio(audio_data, config)
            x_tensor = x_tensor.unsqueeze(0).to(device)
            
            # Predict
            with torch.inference_mode():
                predicted_logit = model(x_tensor)
                chord_pred = torch.softmax(predicted_logit, dim=-1).argmax(dim=-1)

            pred_ids = chord_pred[0].cpu().numpy()
            counts = np.bincount(pred_ids, minlength=config.train.model.output)

            # Penalize "N"
            NO_CHORD_IDX = 0 # index of N
            counts[NO_CHORD_IDX] = int(counts[NO_CHORD_IDX] * 0.05)  # reduce its vote weight

            majority_chord = counts.argmax()
            
            # Display
            chord = postprocess.decode(majority_chord, config)
            print(chord)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    Run()