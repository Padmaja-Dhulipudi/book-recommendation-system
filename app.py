# 📚 Bookish Aesthetic Frontend (Streamlit)

Replace your current `app.py` with this complete frontend-focused version.

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="The Book Nook",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------
# AESTHETIC CSS
# ---------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Poppins:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #f5efe6;
    color: #2f2a24;
}

.stApp {
    background: linear-gradient(to bottom, #f7f1e8, #ede0d4);
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Section */
.hero {
    padding: 40px 20px 10px 20px;
    text-align: center;
}

.hero h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    color: #5c3d2e;
    margin-bottom: 0;
    letter-spacing: 1px;
}

.hero p {
    color: #6d5c4d;
    font-size: 1.2rem;
    margin-top: 5px;
}

/* Search Box */
.stSelectbox label {
    color: #4b3832 !important;
    font-weight: 600;
}

div[data-baseweb="select"] {
    background-color: #fffaf3;
    border-radius: 12px;
    border: 2px solid #d6c3b3;
}

/* Button */
.stButton>button {
    background-color: #6f4e37;
    color: white;
    border-radius: 12px;
    padding: 12px 28px;
    border: none;
    font-size: 16px;
    transition: all 0.3s ease;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #5c3d2e;
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

/* Section Title */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    margin-top: 40px;
    margin-bottom: 20px;
    color: #4b3832;
    border-left: 6px solid #6f4e37;
    padding-left: 12px;
}

/* Book Card */
.book-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    transition: all 0.4s ease;
    height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(111,78,55,0.2);
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}

.book-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 15px 30px rgba(0,0,0,0.18);
    background: #fff8f0;
}

.book-icon {
    font-size: 50px;
    margin-bottom: 10px;
}

.book-title {
    font-size: 16px;
    font-weight: 600;
    color: #3e2c23;
}

.book-sub {
    font-size: 13px;
    color: #7a6758;
    margin-top: 8px;
}

/* Trending Shelf */
.shelf {
    background: rgba(111,78,55,0.08);
    border-radius: 20px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD DATA
# ---------------------------------
@st.cache_data

def load_data():
    df = pd.read_csv("books.csv")

    df.rename(columns={
        'Book-Title': 'title',
        'User-ID': 'user_id',
        'Book-Rating': 'rating'
    }, inplace=True)

    df.dropna(inplace=True)

    return df


data = load_data()

# ---------------------------------
# BUILD MODEL
# ---------------------------------
@st.cache_resource

def build_model(df):
    pivot = df.pivot_table(
        index='title',
        columns='user_id',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(pivot)

    return pivot, similarity


pivot, similarity = build_model(data)
pivot.index = pivot.index.str.lower()

# ---------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------
def recommend(book_name):
    book_name = book_name.strip().lower()

    if book_name not in pivot.index:
        return []

    index = np.where(pivot.index == book_name)[0][0]
    distances = similarity[index]

    books = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:9]

    return [pivot.index[i[0]].title() for i in books]

# ---------------------------------
# HERO SECTION
# ---------------------------------
st.markdown("""
<div class="hero">
    <h1>📚 The Book Nook</h1>
    <p>Find your next comforting read in a world of stories ✨</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# SEARCH SECTION
# ---------------------------------
book_list = sorted([b.title() for b in pivot.index.tolist()])

selected_book = st.selectbox(
    "Choose a book you love:",
    book_list
)

# ---------------------------------
# RECOMMEND BUTTON
# ---------------------------------
if st.button("✨ Recommend Me Books"):

    recommendations = recommend(selected_book)

    if recommendations:

        st.markdown(
            '<div class="section-title">Because you enjoyed this book...</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)

        for i, book in enumerate(recommendations):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-icon">📖</div>
                    <div class="book-title">{book}</div>
                    <div class="book-sub">A beautifully curated recommendation</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------
# TRENDING BOOKS
# ---------------------------------
st.markdown(
    '<div class="section-title">Trending on the Shelf 🔥</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="shelf">', unsafe_allow_html=True)

random_books = np.random.choice(book_list, min(8, len(book_list)), replace=False)

cols = st.columns(4)

for i, book in enumerate(random_books):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="book-card">
            <div class="book-icon">☕</div>
            <div class="book-title">{book}</div>
            <div class="book-sub">Trending among readers</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("""
<br><br>
<center>
<p style='color:#7a6758;'>Made with 🤎 for book lovers</p>
</center>
""", unsafe_allow_html=True)
```

---

# 📁 Required Files

```text
book-recommendation-system/
│
├── app.py
├── books.csv
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# requirements.txt

```txt
streamlit==1.32.0
pandas==2.1.1
numpy==1.26.4
scikit-learn==1.3.2
```

---

# runtime.txt

```txt
python-3.12
```

---

# README.md

````md
# 📚 The Book Nook

A cozy, aesthetic, Netflix-inspired book recommendation web app built using Streamlit.

## ✨ Features

- Beautiful bookish aesthetic UI
- Personalized recommendations
- Trending shelf section
- Hover animations
- Fast recommendation engine using cosine similarity

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
````

## 🌐 Deploy

Deploy easily using Streamlit Cloud.

## 🤎 Designed For

Readers, students, and book lovers who enjoy discovering their next favorite read.

```
```
