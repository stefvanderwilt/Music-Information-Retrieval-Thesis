"""
Shared pipeline configuration for Jamendo manifests, audio source paths, and feature output.
Defines dataset inputs, SSD audio location, extracted features CSV output, and sample settings used by Path A feature extraction and downstream Path B preprocessing.
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_DIR = CODE_DIR / "mtg-jamendo-dataset" / "data"
MOOD_TSV = MANIFEST_DIR / "autotagging_moodtheme.tsv"
GENRE_TSV = MANIFEST_DIR / "autotagging_genre.tsv"

SSD_AUDIO_DIR = Path("/Volumes/StefSD/jamendo_audio")

AUDIO_EXTENSION_SUFFIX = ".low.mp3"

FEATURES_DIR = CODE_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = FEATURES_DIR / "extracted_4pillars.csv"

SAMPLE_RATE = 22050
DURATION_SECONDS = 30.0
