import soundfile as sf
import pyloudnorm as pyln

TARGET_LUFS = -23.0  # EBU R128 standard

def normalize_loudness(input_path, output_path, target_lufs=TARGET_LUFS):
    audio, sr = sf.read(input_path)
    
    meter = pyln.Meter(sr)  # creates BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    
    # loudness normalize
    audio_normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    
    sf.write(output_path, audio_normalized, sr)
    print(f"{input_path}: {loudness:.1f} LUFS → {target_lufs} LUFS")

import os
from pathlib import Path

INPUT_DIRS = ["estimated_sources", "references"]
OUTPUT_BASE = "stimuli_normalized"

for input_dir in INPUT_DIRS:
    for wav_path in Path(input_dir).rglob("*.wav"):
        relative = wav_path.relative_to(input_dir)
        out_path = Path(OUTPUT_BASE) / input_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_loudness(str(wav_path), str(out_path))