# GD Speaker Analysis

Multi-speaker audio analysis tool for group discussions — separates overlapping speakers and analyzes each speaker's voice frequency/pitch characteristics.

## Components
- `pipeline_diarization_separation_analysis.ipynb` — Core ML pipeline (Google Colab): speaker diarization (pyannote.audio), targeted speaker separation (SpeechBrain SepFormer-WHAMR), and per-speaker frequency analysis (librosa)
- `app.py` — Interactive Streamlit dashboard for visualizing results: speaking time analytics, interruption detection, and animated frequency visualization

## How to run

### 1. Run the pipeline (Google Colab)
Open `pipeline_diarization_separation_analysis.ipynb` in Google Colab, add your own Hugging Face token (via Colab Secrets, key named `HF_TOKEN`), and run all cells. This generates:
- Diarization output JSON
- Separated speaker audio files

### 2. Run the dashboard (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload the generated audio and JSON files through the dashboard sidebar to view analytics.

## Tech stack
pyannote.audio, SpeechBrain, librosa, Streamlit, Plotly

## Features
- Speaker diarization with overlap detection
- Targeted 2-speaker source separation on overlapping segments
- Per-speaker pitch, spectral centroid, and energy analysis
- Speaking-time percentage and interruption analytics
- Real-time animated frequency visualization