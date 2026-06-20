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

sample_rate = 44100
frames = []

def callback(indata, frame_count, time_info, status):
    frames.append(indata.copy())

stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback)
with stream:
    input("Recording... press Enter to stop.\n")

recording = np.concatenate(frames, axis=0)
write("output.wav", sample_rate, recording)

#audio_bytes = sys.stdin.buffer.read()

tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
try:
    print(f"[1] Temp-Datei erstellt: {tmp.name}")

    with open("output.wav", "rb") as f:
        audio_bytes = f.read()
        print(f"[2] output.wav gelesen, {len(audio_bytes)} Bytes")

        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        print(f"[3] Bytes in Temp-Datei geschrieben")

        print("[4] Lade Whisper-Modell...")
        model = whisper.load_model("small")
        print("[5] Modell geladen")

        print(f"[6] Starte Transkription von {tmp.name}")
        result = model.transcribe(tmp.name, language="en", verbose=False)
        print("[7] Transkription abgeschlossen")

        segments = [
            {"start": round(seg["start"], 2),  # type: ignore
             "end": round(seg["end"], 2),  # type: ignore
             "text": seg["text"].strip()}  # type: ignore
            for seg in result["segments"]
        ]
        print(f"[8] {len(segments)} Segmente extrahiert")

        print(json.dumps({
            "text": result["text"].strip(),  # type: ignore
            "segments": segments
        }))
        toll = json.dumps({
            "text": result["text"].strip(),  # type: ignore
            "segments": segments
        })
        print(f"[9] {len(segments)} fertig")

except Exception as e:
    print(f"FEHLER: {e}")
    traceback.print_exc()

finally:
    import os
    if os.path.exists(tmp.name):
        #os.unlink(tmp.name)
        print("[10] Temp-Datei gelöscht")# delete temp file manually

i=3
