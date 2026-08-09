import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
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
        padding: 50px 20px 35px 20px;
        text-align: center;
    }

    .hero-title {
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    .book-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        min-height: 190px;
        margin-bottom: 20px;
        box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        text-align: center;
    }

    .book-icon {
        font-size: 45px;
        margin-bottom: 12px;
    }

    .book-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .book-sub {
        font-size: 13px;
        color: #777;
    }

    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        min-height: 150px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
        text-align: center;
    }

    .feature-icon {
        font-size: 35px;
    }

    .feature-title {
        font-weight: 700;
        font-size: 18px;
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

    df.rename(
        columns={
            "Book-Title": "title",
            "User-ID": "user_id",
            "Book-Rating": "rating"
        },
        inplace=True
    )

    df.dropna(subset=["title", "user_id", "rating"], inplace=True)

    return df


data = load_data()


# ==========================================
# BUILD RECOMMENDATION MODEL
# ==========================================

@st.cache_resource
def build_model(df):

    pivot = df.pivot_table(
        index="title",
        columns="user_id",
        values="rating"
    ).fillna(0)

    similarity = cosine_similarity(pivot)

    return pivot, similarity


pivot, similarity = build_model(data)

# Normalize titles for searching
pivot.index = pivot.index.astype(str).str.lower()


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(book_name, number_of_books=8):

    book_name = book_name.strip().lower()

    if book_name not in pivot.index:
        return []

    index = np.where(pivot.index == book_name)[0][0]

    distances = similarity[index]

    books = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:number_of_books + 1]

    recommendations = []

    for book_index, score in books:

        recommendations.append({
            "title": pivot.index[book_index].title(),
            "score": round(float(score) * 100, 1)
        })

    return recommendations


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        📚 The Book Nook
    </div>

    <div class="hero-subtitle">
        Discover your next favorite book with intelligent recommendations.
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# SEARCH SECTION
# ==========================================

st.markdown(
    '<div class="section-title">🔎 Find Your Next Read</div>',
    unsafe_allow_html=True
)

book_list = sorted(
    [book.title() for book in pivot.index.tolist()]
)

selected_book = st.selectbox(
    "Choose a book you love:",
    book_list
)


# ==========================================
# RECOMMENDATION BUTTON
# ==========================================

if st.button("✨ Recommend Me Books", use_container_width=True):

    recommendations = recommend(selected_book)

    if recommendations:

        st.markdown(
            '<div class="section-title">📖 Because you enjoyed this book...</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)

        for i, recommendation in enumerate(recommendations):

            with cols[i % 4]:

                st.markdown(
                    f"""
                    <div class="book-card">

                        <div class="book-icon">
                            📖
                        </div>

                        <div class="book-title">
                            {recommendation["title"]}
                        </div>

                        <div class="book-sub">
                            Similarity Score:
                            <strong>
                                {recommendation["score"]}%
                            </strong>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.warning(
            "Sorry bro 😅 We couldn't find recommendations for this book."
        )


# ==========================================
# TRENDING BOOKS
# ==========================================

st.markdown(
    '<div class="section-title">🔥 Trending on the Shelf</div>',
    unsafe_allow_html=True
)

random_books = np.random.choice(
    book_list,
    min(8, len(book_list)),
    replace=False
)

cols = st.columns(4)

for i, book in enumerate(random_books):

    with cols[i % 4]:

        st.markdown(
            f"""
            <div class="book-card">

                <div class="book-icon">
                    ☕
                </div>

                <div class="book-title">
                    {book}
                </div>

                <div class="book-sub">
                    Popular among readers
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
        "🧹",
        "Data Processing",
        "Cleans and prepares book-rating data for recommendation."
    ),
    (
        "🧠",
        "Similarity Analysis",
        "Uses cosine similarity to measure relationships between books."
    ),
    (
        "✨",
        "Top-N Recommendations",
        "Returns books with the highest similarity to your selected title."
    )
]

for i, (icon, title, text) in enumerate(features):

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
    "🐍 Python   •   🐼 Pandas   •   🔢 NumPy   •   🤖 Scikit-learn   •   🎨 Streamlit"
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
