import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")


st.markdown("""
<style>

/* Background */
body {
    background-color: #141414;
    color: white;
}

/* Main Title */
.title {
    font-size: 3.2rem;
    font-weight: bold;
    color: #E50914;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 1.2rem;
    color: #bbb;
    margin-bottom: 30px;
}

/* Book Card */
.book-card {
    background-color: #222;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    transition: all 0.3s ease-in-out;
    cursor: pointer;
}

/* Hover Effect */
.book-card:hover {
    transform: scale(1.12);
    background-color: #333;
    box-shadow: 0 12px 30px rgba(229, 9, 20, 0.7);
}

/* Book Title */
.book-card h4 {
    font-size: 16px;
    transition: color 0.3s;
}

/* Hover Title Glow */
.book-card:hover h4 {
    color: #E50914;
}

/* Section Headings */
.section-title {
    font-size: 1.5rem;
    margin-top: 30px;
    margin-bottom: 15px;
    color: white;
}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    df.dropna(inplace=True)

   
    df.rename(columns={
        'Book-Title': 'title',
        'User-ID': 'user_id',
        'Book-Rating': 'rating'
    }, inplace=True)

    return df

data = load_data()


@st.cache_resource
def build_model(df):
    pivot = df.pivot_table(index='title', columns='user_id', values='rating')
    pivot.fillna(0, inplace=True)

    similarity = cosine_similarity(pivot)
    return pivot, similarity

pivot, similarity = build_model(data)
pivot.index = pivot.index.str.lower()


def recommend(book_name):
    book_name = book_name.strip().lower()

    if book_name not in pivot.index:
        return []

    index = np.where(pivot.index == book_name)[0][0]
    distances = similarity[index]

    books_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    return [pivot.index[i[0]].title() for i in books_list]


st.markdown('<div class="title">📚 RECOMMENDED BOOKS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your next favorite book 🍿</div>', unsafe_allow_html=True)


book_list = [b.title() for b in pivot.index.tolist()]
selected_book = st.selectbox("🔍 Search a book:", book_list)


if st.button("🎯 Recommend"):
    results = recommend(selected_book)

    if results:
        st.markdown('<div class="section-title">🎬 Because you liked this...</div>', unsafe_allow_html=True)

        cols = st.columns(5)

        for i, book in enumerate(results):
            with cols[i % 5]:
                st.markdown(f"""
                <div class="book-card">
                    <h4>{book}</h4>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("No recommendations found.")

st.markdown('<div class="section-title">🔥 Trending Books</div>', unsafe_allow_html=True)

random_books = np.random.choice(book_list, 10)

cols = st.columns(5)

for i, book in enumerate(random_books):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="book-card">
            <h4>{book}</h4>
        </div>
        """, unsafe_allow_html=True)
