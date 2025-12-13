import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Text Preprocessing ----------------
def tokenization(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z\s]", "", txt)
    return txt


# ---------------- Build Model ----------------
def build_similarity_model(csv_path: str):
    """
    Builds TF-IDF matrix only (NO full similarity matrix).
    Safe for large datasets and cloud deployment.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"song", "artist", "text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["processed_text"] = df["text"].apply(tokenization)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2),
        min_df=5
    )

    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

    return df, tfidf_matrix


# ---------------- Recommendation ----------------
def recommend(song_name, df, tfidf_matrix, top_n=5):
    """
    Computes cosine similarity on-the-fly for one song only.
    """
    if song_name not in df["song"].values:
        return []

    idx = df.index[df["song"] == song_name][0]

    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][1 : top_n + 1]
    return df.iloc[top_indices]["song"].tolist()
