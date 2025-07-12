# ✅ Final Karmonic DJ Insight Bot (Upgraded Version)
# Author: Akshay Surti (Karmonic)

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cohere
import requests
import librosa
import librosa.display
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os
import base64
import plotly.graph_objects as go
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

# --- Config ---
st.set_page_config(page_title="Karmonic DJ Bot", layout="wide")

# --- Load secrets ---
cohere_key = st.secrets["COHERE_API_KEY"]
spotipy_id = st.secrets["SPOTIPY_CLIENT_ID"]
spotipy_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
co = cohere.Client(cohere_key)

# --- Logo display ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("logo.png")
st.markdown(f"""
<style>
.karmonic-header {{
    display: flex;
    align-items: center;
    gap: 20px;
}}
.karmonic-logo {{
    width: 100px;
    height: 100px;
    border-radius: 16px;
    transition: all 0.4s ease-in-out;
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.1);
}}
.karmonic-logo:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 20px 8px #ff00ff, 0 0 40px 14px #00ffff;
}}
.karmonic-title h1 {{
    font-size: 36px;
    font-weight: 800;
    color: white;
}}
.karmonic-title h4 {{
    font-weight: 400;
    color: #cccccc;
}}
</style>
<div class="karmonic-header">
    <img src="data:image/png;base64,{logo_base64}" class="karmonic-logo" alt="Karmonic Logo">
    <div class="karmonic-title">
        <h1>🎧 Karmonic DJ Insight Bot</h1>
        <h4>Feel the Frequency, Understand the Vibe</h4>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_dataset.csv")

df = load_data()
track_names = df["track_name"] + " - " + df["artist_name"]

# --- Dataset Track ---
selected_track = st.selectbox("🎵 Choose a Track", track_names)
track_index = track_names[track_names == selected_track].index[0]
track = df.loc[track_index]
track_id = track["track_id"]

track_features = {
    "valence": track["valence"],
    "energy": track["energy"],
    "danceability": track["danceability"],
    "tempo": track["tempo"],
    "acousticness": track["acousticness"],
    "instrumentalness": track["instrumentalness"]
}

# --- Album Art + Embed ---
def fetch_album_thumb(track_id):
    try:
        url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{track_id}"
        response = requests.get(url)
        return response.json()["thumbnail_url"]
    except:
        return None

album_url = fetch_album_thumb(track_id)
col1, col2 = st.columns([1, 2])
with col1:
    if album_url:
        st.image(album_url, caption="Album Art", width=200)
with col2:
    st.markdown(f"### Now Playing: *{track['track_name']}* by *{track['artist_name']}*")
    st.components.v1.html(
        f"""<iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/track/{track_id}" 
        width="100%" height="80" frameBorder="0" allowtransparency="true" allow="encrypted-media">
        </iframe>""", height=100)

# --- Interactive Radar Chart (Dataset Only) ---
st.markdown("### 🕸️ Audio Profile Radar")
features = {
    "Energy": track_features["energy"],
    "Danceability": track_features["danceability"],
    "Valence": track_features["valence"],
    "Acousticness": track_features["acousticness"],
    "Instrumentalness": track_features["instrumentalness"],
    "Tempo (scaled)": track_features["tempo"] / 250
}
categories = list(features.keys()) + [list(features.keys())[0]]
values = list(features.values()) + [list(features.values())[0]]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Audio Profile',
    line=dict(color='magenta'),
    fillcolor='rgba(255, 0, 255, 0.2)',
    hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
))
fig.update_layout(
    polar=dict(
        bgcolor="#0d0d0d",
        radialaxis=dict(visible=False),
        angularaxis=dict(
            tickfont=dict(size=11, color="#CCCCCC"),
            linecolor="#00ffff"
        )
    ),
    showlegend=False,
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    margin=dict(t=10, b=10, l=10, r=10),
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# --- Mood Summary ---
if st.button("🧠 Describe Mood with AI", key="mood_button_dataset"):
    prompt = f"Given a song with these values: {track_features}, describe the emotion, vibe, and ideal listening moment."
    response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
    st.markdown("### 🧠 Mood Summary:")
    st.success(response.generations[0].text.strip())

# --- Audio Upload + Visuals ---
st.markdown("---")
st.markdown("### 🌈 Audio Waveform with Emotion Overlay")

uploaded_file = st.file_uploader("🎵 Upload a .mp3 or .wav file", type=["mp3", "wav"], key="wave_audio")

if uploaded_file is not None:
    # --- Save uploaded file ---
    tmp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if tmp_path:
        try:
            y, sr = librosa.load(tmp_path, duration=60)
            duration = librosa.get_duration(y=y, sr=sr)

            # --- 🌊 Plotly Waveform ---
            st.markdown("#### 🌊 Real-Time Animated Waveform")
            time_series = np.linspace(0, duration, len(y))
            fig_wave = go.Figure(data=go.Scatter(
                x=time_series,
                y=y,
                mode='lines',
                line=dict(color='magenta', width=1),
                name="Waveform"
            ))
            fig_wave.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                margin=dict(t=10, b=10, l=10, r=10),
                height=250,
                template="plotly_dark",
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#0d0d0d",
                xaxis=dict(color="white"),
                yaxis=dict(color="white"),
            )
            st.plotly_chart(fig_wave, use_container_width=True)

            # --- 🎚️ EQ Pulse ---
            st.markdown("#### 🎚️ EQ Pulse Visualizer")
            D = np.abs(librosa.stft(y, n_fft=1024))
            eq_mean = D.mean(axis=1)
            eq_bins = eq_mean[:60]
            fig_eq, ax_eq = plt.subplots(figsize=(12, 2), dpi=100)
            ax_eq.bar(np.arange(len(eq_bins)), eq_bins, color=plt.cm.viridis(np.linspace(0, 1, len(eq_bins))), width=0.7)
            ax_eq.set_facecolor("#0d0d0d")
            fig_eq.patch.set_facecolor("#0d0d0d")
            ax_eq.set_xticks([])
            ax_eq.set_yticks([])
            for spine in ax_eq.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_eq)

            # --- 🕸️ Radar Chart ---
            st.markdown("### 🕸️ Audio Profile Radar ")

            rms = np.mean(librosa.feature.rms(y=y))
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

            uploaded_features = {
                "Energy": float(rms),
                "Danceability": float(tempo / 200),
                "Valence": float(np.mean(spectral_centroid) / np.max(spectral_centroid)),
                "Acousticness": float(zcr),
                "Instrumentalness": float(1 - np.mean(spectral_bandwidth) / np.max(spectral_bandwidth)),
                "Tempo (scaled)": float(tempo / 250)
            }

            for key in uploaded_features:
                uploaded_features[key] = min(max(uploaded_features[key], 0), 1)

            radar_keys = list(uploaded_features.keys()) + [list(uploaded_features.keys())[0]]
            uploaded_values = list(uploaded_features.values()) + [list(uploaded_features.values())[0]]
            dataset_values = list(track_features.values()) + [list(track_features.values())[0]]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatterpolar(
                r=uploaded_values,
                theta=radar_keys,
                fill='toself',
                name='Your Upload',
                line=dict(color='aqua'),
                fillcolor='rgba(0, 255, 255, 0.3)',
                hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
            ))
            fig_compare.add_trace(go.Scatterpolar(
                r=dataset_values,
                theta=radar_keys,
                fill='toself',
                name='Spotify Track',
                line=dict(color='magenta'),
                fillcolor='rgba(255, 0, 255, 0.2)',
                hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
            ))
            fig_compare.update_layout(
                polar=dict(
                    bgcolor="#0d0d0d",
                    radialaxis=dict(visible=False),
                    angularaxis=dict(
                        tickfont=dict(size=11, color="#CCCCCC"),
                        linecolor="#00ffff"
                    )
                ),
                showlegend=True,
                legend=dict(font=dict(color="white")),
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#0d0d0d",
                margin=dict(t=10, b=10, l=10, r=10),
                height=500
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # --- 🤖 AI Mood Summary ---
            st.markdown("### 🤖 AI Mood & Listening Scenario")
            mood_prompt = f"Given a song with these audio features: {uploaded_features}, describe the overall emotion, energy, and when someone should ideally listen to it."
            mood_response = co.generate(model="command-r-plus", prompt=mood_prompt, max_tokens=300)
            st.success(mood_response.generations[0].text.strip())

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
