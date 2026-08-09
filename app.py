import streamlit as st
import pandas as pd
import numpy as np
import ast
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
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #faf8f4;
}

.hero {
    padding: 45px 20px 30px 20px;
    text-align: center;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 19px;
    color: #666;
    margin-top: 8px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 20px;
}

.book-card {
    background: white;
    padding: 15px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
    min-height: 390px;
}

.book-card img {
    width: 100%;
    height: 240px;
    object-fit: contain;
    border-radius: 10px;
}

.book-title {
    font-size: 17px;
    font-weight: 700;
    margin-top: 12px;
}

.book-author {
    font-size: 14px;
    color: #666;
    margin-top: 5px;
}

.book-rating {
    font-size: 14px;
    margin-top: 8px;
}

.book-score {
    font-size: 13px;
    color: #777;
    margin-top: 5px;
}

.feature-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    min-height: 160px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
}

.feature-icon {
    font-size: 35px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
}

.feature-text {
    color: #777;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #888;
    padding: 40px 0 20px 0;
}

</style>
""", unsafe_allow_html=True)


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

    df["authors"] = df["authors"].astype(str)

    df["title"] = df["title"].astype(str)

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
# CREATE FEATURES
# ==========================================

@st.cache_resource
def build_model(df):

    df = df.copy()

    df["features"] = (
        df["title"] + " " +
        df["authors"] + " " +
        df["original_title"].fillna("").astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["features"]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return similarity_matrix


similarity_matrix = build_model(data)


# ==========================================
# IMAGE URL CLEANER
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

    index = matches[0]

    similarity_scores = similarity_matrix[index]

    similar_indices = (
        np.argsort(similarity_scores)[::-1]
    )

    similar_indices = [
        i for i in similar_indices
        if i != index
    ][:number_of_books]

    recommendations = data.iloc[
        similar_indices
    ].copy()

    recommendations["similarity"] = [
        similarity_scores[i] * 100
        for i in similar_indices
    ]

    return recommendations


# ==========================================
# HERO
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        📚 The Book Nook
    </div>

    <div class="hero-subtitle">
        Discover your next favorite book with AI-powered recommendations.
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# SEARCH
# ==========================================

st.markdown(
    '<div class="section-title">🔎 Find Your Next Read</div>',
    unsafe_allow_html=True
)

book_list = sorted(
    data["title"].tolist()
)

selected_book = st.selectbox(
    "Choose a book you love:",
    book_list
)


# ==========================================
# RECOMMEND
# ==========================================

if st.button(
    "✨ Recommend Me Books",
    use_container_width=True
):

    recommendations = recommend(
        selected_book
    )

    if not recommendations.empty:

        st.markdown(
            '<div class="section-title">📖 You Might Also Like</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)

        for i, (_, book) in enumerate(
            recommendations.iterrows()
        ):

            image_url = get_image_url(
                book["image_url"]
            )

            with cols[i % 4]:

                st.markdown(
                    f"""
                    <div class="book-card">

                        <img
                            src="{image_url}"
                            onerror="this.style.display='none';"
                        >

                        <div class="book-title">
                            {book["title"]}
                        </div>

                        <div class="book-author">
                            ✍️ {book["authors"]}
                        </div>

                        <div class="book-rating">
                            ⭐ {book["average_rating"]}
                            &nbsp; • &nbsp;
                            {int(book["ratings_count"]):,} ratings
                        </div>

                        <div class="book-score">
                            🤖 Similarity:
                            {book["similarity"]:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==========================================
# POPULAR BOOKS
# ==========================================

st.markdown(
    '<div class="section-title">🔥 Popular on the Shelf</div>',
    unsafe_allow_html=True
)

popular_books = data.sort_values(
    "ratings_count",
    ascending=False
).head(8)

cols = st.columns(4)

for i, (_, book) in enumerate(
    popular_books.iterrows()
):

    image_url = get_image_url(
        book["image_url"]
    )

    with cols[i % 4]:

        st.markdown(
            f"""
            <div class="book-card">

                <img
                    src="{image_url}"
                    onerror="this.style.display='none';"
                >

                <div class="book-title">
                    {book["title"]}
                </div>

                <div class="book-author">
                    ✍️ {book["authors"]}
                </div>

                <div class="book-rating">
                    ⭐ {book["average_rating"]}
                </div>

                <div class="book-score">
                    🔥 {int(book["ratings_count"]):,} ratings
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# HOW IT WORKS
# ==========================================

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

cols = st.columns(3)

features = [

    (
        "📊",
        "Feature Extraction",
        "Book titles, authors and original titles are converted into meaningful text features."
    ),

    (
        "🧠",
        "TF-IDF + Cosine Similarity",
        "The model measures how similar books are based on their textual features."
    ),

    (
        "✨",
        "Top Recommendations",
        "The system returns the books with the highest similarity scores."
    )

]

for i, (
    icon,
    title,
    text
) in enumerate(features):

    with cols[i]:

        st.markdown(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    {icon}
                </div>

                <div class="feature-title">
                    {title}
                </div>

                <div class="feature-text">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TECH STACK
# ==========================================

st.markdown(
    '<div class="section-title">🛠️ Technology Stack</div>',
    unsafe_allow_html=True
)

st.write(
    "🐍 Python   •   🐼 Pandas   •   🔢 NumPy   •   "
    "🤖 Scikit-learn   •   🎨 Streamlit"
)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    Made with ❤️ and Python

    <br>

    The Book Nook • AI-powered book discovery

</div>
""", unsafe_allow_html=True)
