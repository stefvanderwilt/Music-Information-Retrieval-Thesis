"""
Validates Jamendo TSV loading and SSD audio access before extraction.
Checks that the Path A extraction inputs are available and that a sample track can be loaded, ensuring the feature pipeline is operational.
"""

import pandas as pd
import librosa
import sys
from pathlib import Path

from config import MOOD_TSV, SSD_AUDIO_DIR, SAMPLE_RATE, DURATION_SECONDS, AUDIO_EXTENSION_SUFFIX

print("\n--- PIPELINE SANITY CHECK ---\n")

def load_jamendo_tsv(tsv_path):
    rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip('\n').split('\t')
            if len(parts) < 5:
                continue
            rows.append({
                'TRACK_ID': parts[0],
                'ARTIST_ID': parts[1],
                'ALBUM_ID': parts[2],
                'PATH': parts[3],
                'DURATION': float(parts[4]) if parts[4] else 0.0,
                'TAGS': ','.join(parts[5:]),
            })
    return pd.DataFrame(rows)

try:
    df = load_jamendo_tsv(MOOD_TSV)
    print(f"Success: Loaded metadata! Total tracks: {len(df)}")
except Exception as e:
    print(f"Error loading spreadsheet: {e}")
    sys.exit()

if SSD_AUDIO_DIR.exists():
    print(f"Success: Found external SSD directory at: {SSD_AUDIO_DIR}")
else:
    print(f"Error: Cannot see external SSD at {SSD_AUDIO_DIR}.")
    sys.exit()

print("\nAttempting to load a sample track via Librosa...")
try:
    sample_relative_path = df.iloc[0]['PATH']
    adjusted_relative_path = sample_relative_path.replace('.mp3', AUDIO_EXTENSION_SUFFIX)
    sample_absolute_path = SSD_AUDIO_DIR / adjusted_relative_path

    print(f"Targeting adjusted file: {sample_absolute_path}")

    if not sample_absolute_path.exists():
        print("Error: Audio file still does not exist at that path.")
        sys.exit()

    y, sr = librosa.load(sample_absolute_path, sr=SAMPLE_RATE, duration=5.0)

    print(f"Success: Librosa successfully loaded {len(y)} audio samples.")
    print(f"Sample Rate: {sr}Hz")
    print("\nPipeline operational.\n")
except Exception as e:
    print(f"Error loading audio file: {e}")
