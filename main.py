#pip install -U openai-whisper==20250625
import whisper
import sys
import tempfile
import json
import traceback
import os

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import ssl
import certifi

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

class Hackathon:
    def __init__(self):
        self.sample_rate = 44100
        self.frames = []

        self.recording = self.input_voice_raw()
        write("output.wav", self.sample_rate, self.recording)
        self.word_string = self.Voice2String()
        i=5

    def Voice2String(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            print(f"[1] Temp-Datei erstellt: {tmp.name}")

            with open("output.wav", "rb") as f:
                audio_bytes = f.read()
                tmp.write(audio_bytes)
                tmp.flush()
                tmp.close()
                model = whisper.load_model("small")
                result = model.transcribe(tmp.name, language="en", verbose=False)
                segments = [
                    {"start": round(seg["start"], 2),  # type: ignore
                     "end": round(seg["end"], 2),  # type: ignore
                     "text": seg["text"].strip()}  # type: ignore
                    for seg in result["segments"]
                ]
                print(json.dumps({
                    "text": result["text"].strip(),  # type: ignore
                    "segments": segments
                }))
                output_string = result["text"].strip()

        except Exception as e:
            print(f"FEHLER: {e}")
            traceback.print_exc()

        finally:
            import os
            if os.path.exists(tmp.name):
                 os.unlink(tmp.name)
        return output_string

    def callback(self,indata, frame_count, time_info, status):
        self.frames.append(indata.copy())

    def input_voice_raw(self):
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", callback=self.callback)
        with stream:
            input("Recording... press Enter to stop.\n")

        recording = np.concatenate(self.frames, axis=0)
        return recording

Hackathon = Hackathon()

#audio_bytes = sys.stdin.buffer.read()
