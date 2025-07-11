import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cohere
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import sys

sys.path.append(os.path.abspath("app"))
from prompts import (
    mood_summary_prompt,
    genre_match_prompt,
    spiritual_context_prompt,
    listening_context_prompt,
)

# --- Layout & Style ---
st.set_page_config(page_title="Karmonic DJ Bot", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #f0f0f0;
        }
        .stButton>button {
            background-color: #7f00ff;
            color: white;
            padding: 0.5em 1em;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #ba68c8;
        }
        h1 {
            text-align: center;
            color: orchid;
            font-size: 40px;
        }
        .controls-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-top: 10px;
        }
        .control-button {
            font-size: 18px;
            padding: 6px 14px;
            background-color: #7f00ff;
            color: white;
            border-radius: 6px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Secrets and Data ---
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_dataset.csv")

df = load_data()

# --- Logo ---
logo = Image.open("logo.png")
st.image(logo, width=120)

# --- Song Selection & Filters ---
st.markdown("## 🎧 *Karmonic Insight Bot* — *Feel the Frequency*")

track_names = df["track_name"] + " - " + df["artist_name"]
selected_track = st.selectbox("🔍 Search & Select a Track", track_names)

track_index = track_names[track_names == selected_track].index[0]
track = df.loc[track_index]

track_features = {
    "valence": track["valence"],
    "energy": track["energy"],
    "danceability": track["danceability"],
    "tempo": track["tempo"],
    "acousticness": track.get("acousticness", 0),
    "instrumentalness": track.get("instrumentalness", 0)
}

# --- Album Art ---
def fetch_album_thumb(track_id):
    try:
        url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{track_id}"
        response = requests.get(url)
        return response.json()["thumbnail_url"]
    except:
        return None

album_url = fetch_album_thumb(track["track_id"])

# --- Player Controls ---
spotify_url = f"https://open.spotify.com/embed/track/{track['track_id']}"

col1, col2 = st.columns([1, 2])
with col1:
    if album_url:
        st.image(album_url, caption="Album Art", width=200)
    else:
        st.info("No album art available")

with col2:
    st.markdown("### ▶️ Now Playing:")
    st.markdown(f"**{track['track_name']}** by *{track['artist_name']}*")
    st.components.v1.html(
        f"""
        <iframe style="border-radius:12px" 
            src="{spotify_url}" 
            width="100%" 
            height="80" 
            frameBorder="0" 
            allowtransparency="true" 
            allow="encrypted-media">
        </iframe>
        """,
        height=100,
    )
    st.slider("🔊 Volume (UI only)", 0, 100, 50)

    st.markdown("#### ⏯️ Controls (visual only)")
    col_prev, col_play, col_next = st.columns(3)
    with col_prev: st.button("⏮️ Back")
    with col_play: st.button("⏯️ Play/Pause")
    with col_next: st.button("⏭️ Next")

# --- Radar Chart ---
st.markdown("## 🕸️ Track Audio Profile")

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

fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='orchid', linewidth=2)
ax.fill(angles, values, alpha=0.3, color='orchid')
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
st.pyplot(fig)

# --- AI Insight Buttons (Vertical) ---
st.markdown("## 🧠 AI-Powered Track Insights")

if st.button("🧠 Describe Mood with AI"):
    with st.spinner("Asking Karmonic..."):
        prompt = mood_summary_prompt(track_features)
        response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
        st.markdown("### 🎶 Mood Summary:")
        st.success(response.generations[0].text.strip())

if st.button("🎧 Genre Match"):
    with st.spinner("Analyzing genre vibe..."):
        prompt = genre_match_prompt(track_features)
        response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
        st.markdown("### 🧬 Genre Match Insight:")
        st.info(response.generations[0].text.strip())

if st.button("🧘 Spiritual Context"):
    with st.spinner("Searching for deeper meaning..."):
        prompt = spiritual_context_prompt(track_features)
        response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
        st.markdown("### 🔮 Spiritual Use:")
        st.info(response.generations[0].text.strip())

if st.button("📍 When Should I Listen?"):
    with st.spinner("Feeling the vibe..."):
        prompt = listening_context_prompt(track_features)
        response = co.generate(model="command-r-plus", prompt=prompt, max_tokens=300)
        st.markdown("### 🕰️ Best Listening Moment:")
        st.info(response.generations[0].text.strip())
