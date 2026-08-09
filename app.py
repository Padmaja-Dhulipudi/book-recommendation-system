import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="The Book Nook",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("books.csv")

    required_columns = [
        "title",
        "authors",
        "average_rating",
        "ratings_count",
        "image_url"
    ]

    df = df.dropna(subset=required_columns)

    df = df.drop_duplicates(subset=["title"])

    df["title"] = df["title"].astype(str)
    df["authors"] = df["authors"].astype(str)

    df["original_title"] = (
        df["original_title"]
        .fillna("")
        .astype(str)
    )

    df["average_rating"] = pd.to_numeric(
        df["average_rating"],
        errors="coerce"
    )

    df["ratings_count"] = pd.to_numeric(
        df["ratings_count"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["average_rating", "ratings_count"]
    )

    return df.reset_index(drop=True)


data = load_data()


# ==========================================
# BUILD TF-IDF MODEL
# ==========================================

@st.cache_resource
def build_model(df):

    features = (
        df["title"] + " " +
        df["authors"] + " " +
        df["original_title"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(features)

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = build_model(data)


# ==========================================
# IMAGE URL
# ==========================================

def get_image_url(value):

    if pd.isna(value):
        return ""

    value = str(value)

    urls = re.findall(
        r"https?://[^\s,\]]+",
        value
    )

    if urls:
        return urls[0]

    return ""


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(book_title, number_of_books=8):

    matches = data.index[
        data["title"] == book_title
    ].tolist()

    if not matches:
        return pd.DataFrame()

    book_index = matches[0]

    selected_vector = tfidf_matrix[
        book_index
    ]

    scores = cosine_similarity(
        selected_vector,
        tfidf_matrix
    ).flatten()

    similar_indices = np.argsort(
        scores
    )[::-1]

    similar_indices = [
        index
        for index in similar_indices
        if index != book_index
    ][:number_of_books]

    recommendations = data.iloc[
        similar_indices
    ].copy()

    recommendations["similarity"] = [
        scores[index] * 100
        for index in similar_indices
    ]

    return recommendations


# ==========================================
# HEADER
# ==========================================

st.title("📚 The Book Nook")

st.write(
    "Discover your next favorite book with "
    "AI-powered recommendations."
)

st.divider()


# ==========================================
# SEARCH
# ==========================================

st.header("🔎 Find Your Next Read")

book_list = sorted(
    data["title"].tolist()
)

selected_book = st.selectbox(
    "Choose a book you love:",
    book_list
)


# ==========================================
# RECOMMENDATIONS
# ==========================================

if st.button(
    "✨ Recommend Me Books",
    use_container_width=True
):

    recommendations = recommend(
        selected_book
    )

    if not recommendations.empty:

        st.header("📖 You Might Also Like")

        cols = st.columns(4)

        for i, (_, book) in enumerate(
            recommendations.iterrows()
        ):

            with cols[i % 4]:

                image_url = get_image_url(
                    book["image_url"]
                )

                if image_url:
                    st.image(
                        image_url,
                        use_container_width=True
                    )

                st.subheader(
                    book["title"]
                )

                st.caption(
                    f"✍️ {book['authors']}"
                )

                st.write(
                    f"⭐ {book['average_rating']:.2f}"
                )

                st.caption(
                    f"🤖 Similarity: "
                    f"{book['similarity']:.1f}%"
                )

                st.caption(
                    f"📊 {int(book['ratings_count']):,} ratings"
                )


# ==========================================
# POPULAR BOOKS
# ==========================================

st.divider()

st.header("🔥 Popular on the Shelf")

popular_books = data.sort_values(
    "ratings_count",
    ascending=False
).head(8)

cols = st.columns(4)

for i, (_, book) in enumerate(
    popular_books.iterrows()
):

    with cols[i % 4]:

        image_url = get_image_url(
            book["image_url"]
        )

        if image_url:
            st.image(
                image_url,
                use_container_width=True
            )

        st.subheader(
            book["title"]
        )

        st.caption(
            f"✍️ {book['authors']}"
        )

        st.write(
            f"⭐ {book['average_rating']:.2f}"
        )

        st.caption(
            f"🔥 {int(book['ratings_count']):,} ratings"
        )


# ==========================================
# HOW IT WORKS
# ==========================================

st.divider()

st.header("⚙️ How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Feature Extraction")
    st.write(
        "Book titles, authors and original titles "
        "are converted into text features."
    )

with col2:
    st.subheader("🧠 TF-IDF")
    st.write(
        "TF-IDF converts book information into "
        "numerical vectors for comparison."
    )

with col3:
    st.subheader("✨ Recommendations")
    st.write(
        "Cosine similarity finds books that are "
        "most similar to your selected book."
    )


# ==========================================
# TECH STACK
# ==========================================

st.divider()

st.subheader("🛠️ Tech Stack")

st.write(
    "Python • Pandas • NumPy • Scikit-learn • Streamlit"
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "📚 The Book Nook — AI-powered book discovery"
)
