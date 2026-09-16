import streamlit as st
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import defaultdict
from pydub import AudioSegment
import tempfile
import os
import streamlit.components.v1 as components
import base64

st.set_page_config(page_title="GD Speaker Analysis", layout="wide")
st.title(" Group Discussion Speaker Analysis")

# ---------- Helper functions ----------
def compute_analytics(segments):
    speaker_time = defaultdict(float)
    total_time = 0
    interruptions = defaultdict(int)
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    for i, seg in enumerate(sorted_segs):
        dur = seg['end'] - seg['start']
        speaker_time[seg['speaker']] += dur
        total_time += dur
        if i > 0:
            prev = sorted_segs[i-1]
            if seg['start'] < prev['end'] and seg['speaker'] != prev['speaker']:
                interruptions[seg['speaker']] += 1
    return speaker_time, interruptions, total_time

def analyze_pitch(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    return {
        "avg_pitch": float(np.mean(f0_clean)) if len(f0_clean) else 0,
        "pitch_variance": float(np.var(f0_clean)) if len(f0_clean) else 0,
        "spectral_centroid": float(np.mean(spec_centroid)),
        "energy": float(np.mean(rms))
    }, f0, sr
    
def audio_visualizer(audio_bytes, label="Audio"):
    b64 = base64.b64encode(audio_bytes).decode()
    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    html_code = f"""
    <div style="text-align:center; font-family:sans-serif;">
      <p style="color:#666;">{label}</p>
      <canvas id="canvas_{safe_label}" width="700" height="120" style="background:#0e1117; border-radius:8px;"></canvas>
      <br>
      <audio id="audio_{safe_label}" controls style="margin-top:10px; width:700px;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
      </audio>
    </div>
    <script>
      const audio_{safe_label} = document.getElementById('audio_{safe_label}');
      const canvas_{safe_label} = document.getElementById('canvas_{safe_label}');
      const ctx_{safe_label} = canvas_{safe_label}.getContext('2d');
      const audioCtx_{safe_label} = new (window.AudioContext || window.webkitAudioContext)();
      const source_{safe_label} = audioCtx_{safe_label}.createMediaElementSource(audio_{safe_label});
      const analyser_{safe_label} = audioCtx_{safe_label}.createAnalyser();
      source_{safe_label}.connect(analyser_{safe_label});
      analyser_{safe_label}.connect(audioCtx_{safe_label}.destination);
      analyser_{safe_label}.fftSize = 256;
      const bufferLength_{safe_label} = analyser_{safe_label}.frequencyBinCount;
      const dataArray_{safe_label} = new Uint8Array(bufferLength_{safe_label});

      function draw_{safe_label}() {{
        requestAnimationFrame(draw_{safe_label});
        analyser_{safe_label}.getByteFrequencyData(dataArray_{safe_label});
        ctx_{safe_label}.fillStyle = '#0e1117';
        ctx_{safe_label}.fillRect(0, 0, canvas_{safe_label}.width, canvas_{safe_label}.height);
        const barWidth = (canvas_{safe_label}.width / bufferLength_{safe_label}) * 2.5;
        let x = 0;
        for (let i = 0; i < bufferLength_{safe_label}; i++) {{
          const barHeight = dataArray_{safe_label}[i] * 0.9;
          const hue = 200 + (i / bufferLength_{safe_label}) * 100;
          ctx_{safe_label}.fillStyle = `hsl(${{hue}}, 80%, 60%)`;
          ctx_{safe_label}.fillRect(x, canvas_{safe_label}.height - barHeight, barWidth, barHeight);
          x += barWidth + 1;
        }}
      }}
      draw_{safe_label}();
    </script>
    """
    components.html(html_code, height=250)
# ---------- Sidebar: file upload ----------
st.sidebar.header("Upload")
audio_file = st.sidebar.file_uploader("Group discussion audio (.wav)", type=["wav"])
diarization_json = st.sidebar.file_uploader("Diarization output (.json)", type=["json"])

if audio_file and diarization_json:
    segments = json.load(diarization_json)

    # Filter tiny fragments (same logic as your Colab pipeline)
    min_dur = st.sidebar.slider("Minimum segment duration (s)", 0.1, 2.0, 1.0)
    clean_segments = [s for s in segments if (s['end'] - s['start']) >= min_dur]

    st.subheader(" Speaker Analytics")
    speaker_time, interruptions, total_time = compute_analytics(clean_segments)

    cols = st.columns(len(speaker_time))
    for i, (spk, t) in enumerate(speaker_time.items()):
        pct = (t / total_time) * 100
        with cols[i]:
            st.metric(spk, f"{pct:.1f}%", f"{t:.1f}s speaking")
            st.caption(f"Interruptions caused: {interruptions[spk]}")

    st.subheader(" Diarization Timeline")
    st.dataframe(clean_segments)

    st.subheader(" Original Audio")
    st.audio(audio_file)

    st.subheader(" Per-Speaker Frequency Analysis")
    st.info("Upload individually separated speaker tracks below to see pitch/spectrogram analysis")

    separated_files = st.file_uploader(
        "Separated speaker audio files (upload one or more)",
        type=["wav"], accept_multiple_files=True
    )

    if separated_files:
        for sep_file in separated_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(sep_file.read())
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=None)
            stats, f0, sr = analyze_pitch(y, sr)

            st.markdown(f"**{sep_file.name}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Pitch", f"{stats['avg_pitch']:.0f} Hz")
            c2.metric("Pitch Variance", f"{stats['pitch_variance']:.0f}")
            c3.metric("Spectral Centroid", f"{stats['spectral_centroid']:.0f} Hz")
            c4.metric("Energy (RMS)", f"{stats['energy']:.4f}")

            fig, ax = plt.subplots(figsize=(10, 3))
            times = librosa.times_like(f0, sr=sr)
            ax.plot(times, f0, color='b')
            ax.set_title(f"{sep_file.name} — Pitch Contour")
            ax.set_ylabel("Frequency (Hz)")
            st.pyplot(fig)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            audio_visualizer(audio_bytes, label=sep_file.name)
            os.unlink(tmp_path)

else:
    st.info(" Upload your group discussion audio and diarization JSON to get started")
    st.markdown("""
    **How to use:**
    1. Run your Colab pipeline (diarization → separation → analysis)
    2. Download `diarization_output.json` and your audio files
    3. Upload them here to see the analytics dashboard
    """)