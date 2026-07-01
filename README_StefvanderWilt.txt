This package contains the subset of files used by Stef van der Wilt
for his thesis in Applied Data Science.

Included:
- Source code in `src/`
- Notebook `src/preprocessing.ipynb`
- Jamendo dataset TSV files in `mtg-jamendo-dataset/data/`
- Tag lists in `mtg-jamendo-dataset/data/tags/`
- Statistics files in `mtg-jamendo-dataset/stats/raw_30s_cleantags_50artists/`
- Extracted audio features in `features/extracted_4pillars.csv`

Note:
- The original `raw_30s_cleantags_50artists` audio files from MTG Jamendo are 
not included here because they are too large, but are available openly online.
- This package includes the extracted features derived from those audio files, 
but not the audio files themselves, you can find them in 'features' -> 'extracted_4pillars'.
