"""
Extracts audio features from Jamendo manifest tracks and writes them to extracted_4pillars.csv.
Consumes the Path A audio manifest and SSD audio files, then produces the feature CSV that both Path A training and Path B stacking preprocessing depend on.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

from config import MOOD_TSV, SSD_AUDIO_DIR, OUTPUT_CSV, SAMPLE_RATE, DURATION_SECONDS, AUDIO_EXTENSION_SUFFIX


def load_jamendo_tsv(tsv_path):
    """Robust custom reader to parse the Jamendo TSV structure cleanly."""
    rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip('\n').split('\t')
            if len(parts) < 5:
                continue
            rows.append({
                'TRACK_ID': parts[0],
                'PATH': parts[3],
                'DURATION': float(parts[4]) if parts[4] else 0.0,
            })
    return pd.DataFrame(rows)


def extract_4_pillars(audio_path, sr=22050, duration=30.0, offset=0.0):
    """Loads an audio file from a targeted offset and summarizes the 4 pillars."""
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=offset)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_val = tempo[0] if isinstance(tempo, (np.ndarray, list)) else tempo

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_var = np.var(chroma, axis=1)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    features = {
        'tempo': float(tempo_val),
        'rms_mean': float(rms_mean),
        'rms_var': float(rms_var),
    }

    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(mfcc_mean[i])
        features[f'mfcc_{i+1}_var'] = float(mfcc_var[i])

    for i in range(12):
        features[f'chroma_{i+1}_mean'] = float(chroma_mean[i])
        features[f'chroma_{i+1}_var'] = float(chroma_var[i])

    return features


def run_master_pipeline(df_manifest):
    """Runs the full extraction systematically for the given manifest with direct disk writes."""
    print("Initializing master extraction engine...")

    already_done = set()
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV, usecols=['TRACK_ID'])
            already_done = set(df_existing['TRACK_ID'].astype(str).tolist())
            print(f"Found existing file with {len(already_done)} tracks. Resuming seamlessly...")
        except Exception:
            print("Output file was empty or unreadable. Starting fresh.")

    print(f"Total target capacity for this run: {len(df_manifest)} tracks.")

    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Processing Audio"):
        track_id = str(row['TRACK_ID'])

        if track_id in already_done:
            continue

        adjusted_relative_path = row['PATH'].replace(".mp3", AUDIO_EXTENSION_SUFFIX)
        absolute_audio_path = SSD_AUDIO_DIR / adjusted_relative_path

        try:
            if not absolute_audio_path.exists():
                continue
        except OSError as e:
            print(f"\nSSD I/O connection stale on {track_id}: {e}.")
            print("Waiting 15 seconds to let macOS reset the drive mount...")
            time.sleep(15.0)
            continue

        try:
            total_duration = float(row.get('DURATION', 0.0))
            if total_duration > 30.0:
                start_offset = (total_duration / 2.0) - 15.0
            else:
                start_offset = 0.0

            features = extract_4_pillars(
                absolute_audio_path,
                sr=SAMPLE_RATE,
                duration=DURATION_SECONDS,
                offset=start_offset,
            )
            features['TRACK_ID'] = track_id

            df_row = pd.DataFrame([features])
            cols = ['TRACK_ID'] + [c for c in df_row.columns if c != 'TRACK_ID']
            df_row = df_row[cols]

            if not OUTPUT_CSV.exists():
                df_row.to_csv(OUTPUT_CSV, index=False)
            else:
                df_row.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

            time.sleep(0.1)

        except Exception as e:
            print(f"\nCorrupt or unreadable track skipped {track_id}: {e}")
            continue

    print(f"\nSuccess. Processing batch complete. Data saved cleanly to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    PILOT_MODE = False
    PILOT_LIMIT = 50

    if PILOT_MODE:
        print(f"Pilot mode active: Limiting processing to a maximum of {PILOT_LIMIT} tracks.")
        df_manifest_full = load_jamendo_tsv(MOOD_TSV)

        already_done = set()
        if OUTPUT_CSV.exists():
            try:
                df_existing = pd.read_csv(OUTPUT_CSV, usecols=['TRACK_ID'])
                already_done = set(df_existing['TRACK_ID'].astype(str).tolist())
            except Exception:
                pass

        remaining_tracks = df_manifest_full[~df_manifest_full['TRACK_ID'].astype(str).isin(already_done)]
        df_manifest = remaining_tracks.head(PILOT_LIMIT)

        if len(df_manifest) == 0:
            print("Pilot run complete. 50 tracks have already been successfully extracted.")
            sys.exit(0)
    else:
        print("Production mode active: Targeting entire dataset.")
        df_manifest = load_jamendo_tsv(MOOD_TSV)

    run_master_pipeline(df_manifest)
