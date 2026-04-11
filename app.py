import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Book Recommendation System", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("books.csv")
    except FileNotFoundError:
        st.error("❌ books.csv not found. Please upload it to your repo.")
        st.stop()

    df.dropna(inplace=True)

    df.rename(columns={
        'Book-Title': 'title',
        'User-ID': 'user_id',
        'Book-Rating': 'rating'
    }, inplace=True)

    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

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
    book_name = book_name.lower().strip()

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


st.title("📚 Book Recommendation System")
st.write("Discover your next favorite book!")

book_list = sorted([b.title() for b in pivot.index.tolist()])
selected_book = st.selectbox("🔍 Search a book:", book_list)

if st.button("🎯 Recommend"):
    results = recommend(selected_book)

    if results:
        st.subheader("📖 Recommended Books")

        cols = st.columns(5)
        for i, book in enumerate(results):
            with cols[i % 5]:
                st.markdown(f"""
                <div style="
                    background-color:#262730;
                    padding:15px;
                    border-radius:10px;
                    text-align:center;">
                    <b>{book}</b>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found.")

st.subheader(" Trending Books")

random_books = np.random.choice(book_list, min(10, len(book_list)), replace=False)

cols = st.columns(5)
for i, book in enumerate(random_books):
    with cols[i % 5]:
        st.markdown(f"""
        <div style="
            background-color:#1f1f1f;
            padding:12px;
            border-radius:8px;
            text-align:center;">
            {book}
        </div>
        """, unsafe_allow_html=True)
