import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    df.dropna(inplace=True)
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
    )[1:6]

    return [pivot.index[i[0]].title() for i in books_list]


st.set_page_config(page_title="Book Recommender", layout="centered")

st.title("📚 Book Recommendation System")
st.markdown("Get book suggestions instantly!")

book_input = st.text_input("Enter a book name:")

if st.button("Recommend"):
    if book_input:
        results = recommend(book_input)

        if results:
            st.subheader("📖 Recommended Books:")
            for book in results:
                st.write(f"👉 {book.title()}")
        else:
            st.error("Book not found. Try another title.")
    else:
        st.warning("Please enter a book name.")
