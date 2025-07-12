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

# --- Streamlit page config ---
st.set_page_config(page_title="Karmonic DJ Bot", layout="wide")

# --- Load secrets from Streamlit Cloud ---
cohere_key = st.secrets["COHERE_API_KEY"]
spotipy_id = st.secrets["SPOTIPY_CLIENT_ID"]
spotipy_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
co = cohere.Client(cohere_key)

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_dataset.csv")

# --- Logo ---
logo = Image.open("logo.png")
st.image(logo, width=120)

df = load_data()

# --- Track selection from dataset ---
st.title("🎧 Karmonic DJ Insight Bot")
st.subheader("Feel the Frequency, Understand the Vibe")

track_names = df["track_name"] + " - " + df["artist_name"]
selected_track = st.selectbox("🎵 Choose a Track", track_names)
track_index = track_names[track_names == selected_track].index[0]
track = df.loc[track_index]
track_id = track["track_id"]

# --- Track features ---
track_features = {
    "valence": track["valence"],
    "energy": track["energy"],
    "danceability": track["danceability"],
    "tempo": track["tempo"],
    "acousticness": track.get("acousticness", 0),
    "instrumentalness": track.get("instrumentalness", 0)
}

# --- Album art and Spotify embed ---
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
        f"""
        <iframe style="border-radius:12px" 
            src="https://open.spotify.com/embed/track/{track_id}" 
            width="100%" height="80" frameBorder="0" 
            allowtransparency="true" allow="encrypted-media">
        </iframe>
        """,
        height=100,
    )

# --- Radar Chart from dataset ---
st.markdown("### 🕸️ Audio Profile Radar")
features = {
    "Energy": track_features["energy"],
    "Danceability": track_features["danceability"],
    "Valence": track_features["valence"],
    "Acousticness": track_features["acousticness"],
    "Instrumentalness": track_features["instrumentalness"],
    "Tempo (scaled)": track_features["tempo"] / 250
}
labels = list(features.keys())
values = list(features.values()) + [features["Energy"]]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid', color='orchid')
ax.fill(angles, values, alpha=0.3, color='orchid')
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
st.pyplot(fig)

# --- Mood Summary from dataset ---
if st.button("🧠 Describe Mood with AI", key="mood_button_dataset"):
    prompt = f"Given a song with these values: {track_features}, describe the emotion, vibe, and ideal listening moment."
    response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
    st.markdown("### 🧠 Mood Summary:")
    st.success(response.generations[0].text.strip())

# --- Upload + Analysis ---
st.markdown("## 🎛️ Upload Your Own Audio Track")
uploaded_file = st.file_uploader("🎵 Upload a .mp3 or .wav file", type=["mp3", "wav"])

if uploaded_file:
    with st.spinner("Analyzing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        y, sr = librosa.load(tmp_path, duration=60)

        if y.max() < 0.01:
            st.error("⚠️ Audio is too quiet or silent. Try uploading a louder or clearer file.")
            st.stop()

        features = {
            "Energy": float(np.mean(librosa.feature.rms(y=y))),
            "Danceability": float(np.mean(librosa.feature.tempogram(y=y).mean())),
            "Valence": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "Acousticness": float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "Instrumentalness": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            "Tempo (scaled)": float(librosa.beat.tempo(y=y, sr=sr)[0]) / 250
        }

        max_val = max(max(features.values()), 0.01)
        normalized = {k: round(min(v / max_val, 1.0), 3) for k, v in features.items()}
        st.write("🎛️ Extracted Features (raw):", features)

        # 🔮 AI Mood
        with st.spinner("Feeling the track’s emotion..."):
            mood_prompt = f"This track has these features: {normalized}. Describe the emotional tone in one short sentence."
            vibe_response = co.generate(model="command-r-plus", prompt=mood_prompt, max_tokens=60)
            vibe_text = vibe_response.generations[0].text.strip()
            st.markdown("### 🧠 Mood Vibe:")
            st.success(vibe_text)

        # 🎨 Emotion color mapping
        vibe_color = "orchid"
        vibe_map = {
            "calm": "deepskyblue", "peace": "deepskyblue",
            "happy": "gold", "uplifting": "gold",
            "energetic": "crimson", "hype": "crimson",
            "sad": "mediumpurple", "melancholy": "mediumpurple",
            "dark": "slategray", "emotional": "plum",
            "romantic": "hotpink", "dreamy": "lightskyblue"
        }
        for mood, color in vibe_map.items():
            if mood in vibe_text.lower():
                vibe_color = color
                break

        # 🌊 Waveform visualization
        st.markdown("### 🌈 Audio Waveform (Emotion-Based)")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, alpha=0.65, color=vibe_color, ax=ax)
        ax.set_title(f"🎛️ Emotion Color: {vibe_color}", fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor('#111111')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 🔁 Playback of Uploaded Track
        st.markdown("### ▶️ Play Uploaded Track")
        st.audio(tmp_path, format="audio/wav")

        # 🎛️ Animated EQ Simulation (simulated using amplitude chunks)
        st.markdown("### 🎚️ Simulated EQ Pulse")

        # Slice audio into segments (e.g., 30 bars)
        n_bars = 30
        samples_per_bar = len(y) // n_bars
        amplitudes = [np.mean(np.abs(y[i * samples_per_bar:(i + 1) * samples_per_bar])) for i in range(n_bars)]

        # Bar chart to simulate EQ
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_bars))
        ax.bar(range(n_bars), amplitudes, color=colors)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, n_bars)
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor('#111111')
        st.pyplot(fig)


        # 🎯 AI Detailed Summary
        st.markdown("### 🧠 AI Mood Summary:")
        if st.button("🔍 Describe This Song", key="describe_uploaded_audio"):
            prompt = f"This track has the following audio features: {normalized}. Describe the mood, emotional vibe, and how a DJ might use it in a set."
            response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=250)
            st.success(response.generations[0].text.strip())

# --- DJ Set Recommender from dataset ---
st.markdown("## 🎚️ DJ Set Recommender")
feature_cols = ["valence", "energy", "danceability", "tempo", "acousticness", "instrumentalness"]
df_filtered = df.dropna(subset=feature_cols)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_filtered[feature_cols])
uploaded_scaled = scaler.transform([list(track_features.values())])
similarities = cosine_similarity(uploaded_scaled, scaled_features)[0]
df_filtered["similarity"] = similarities
top_matches = df_filtered.sort_values(by="similarity", ascending=False).head(5)[
    ["track_name", "artist_name", "genre", "similarity"]
]
st.dataframe(top_matches.style.format({"similarity": "{:.3f}"}))
