import streamlit as st
from pathlib import Path
import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from recommender import build_similarity_model, recommend


# ---------------- Spotify Config ----------------
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Spotify credentials not set.")
    st.stop()

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)


def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]

    return "https://i.postimg.cc/0QNxYz4V/social.png"


# ---------------- Load Data & Model ----------------
@st.cache_data(show_spinner=True)
def load_model():
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "spotify_millsongdata.csv"

    if not csv_path.exists():
        st.error("Dataset file not found.")
        st.stop()

    return build_similarity_model(csv_path)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Music Recommender", layout="wide")
st.header("🎵 Music Recommender System")

music, tfidf_matrix = load_model()

selected_song = st.selectbox(
    "Type or select a song",
    music["song"].values
)

if st.button("Show Recommendation"):
    recommended_songs = recommend(
        selected_song,
        music,
        tfidf_matrix,
        top_n=5
    )

    cols = st.columns(len(recommended_songs))
    for col, song in zip(cols, recommended_songs):
        artist = music[music["song"] == song].iloc[0].artist
        with col:
            st.text(song)
            st.image(get_song_album_cover_url(song, artist))
