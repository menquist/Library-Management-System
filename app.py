import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline


st.set_page_config(page_title="Library Management System", layout="centered")

st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa_model

embedder, qa_model = load_models()

# Initial data
if "books" not in st.session_state:
    st.session_state.books = [
        {
            "title": "1984",
            "author": "George Orwell",
            "year": "1949",
            "status": "Available",
            "description": "Dystopian novel set in Airstrip One."
        },
        {
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "year": "1960",
            "status": "Available",
            "description": "Story of racial injustice in the Deep South."
        },
        {
            "title": "A Brief History of Time",
            "author": "Stephen Hawking",
            "year": "1988",
            "status": "Available",
            "description": "Exploration of cosmology for the general reader."
        },
        {
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "year": "1925",
            "status": "Available",
            "description": "Novel about the American dream and excess."
        },
        {
            "title": "The Catcher in the Rye",
            "author": "J. D. Salinger",
            "year": "1951",
            "status": "Available",
            "description": "Holden Caulfield narrates his teenage angst."
        },
    ]

def embed_books():
    texts = [
        f"{b['title']} {b['author']} {b.get('year','')} {b.get('description','')}"
        for b in st.session_state.books
    ]
    return embedder.encode(texts) if texts else np.empty((0, 384))

if "embeddings" not in st.session_state:
    st.session_state.embeddings = embed_books()

st.title("Mini Library Management System")

# Add book form
with st.form("add_form"):
    st.subheader("Add Book")
    col1, col2 = st.columns(2)
    with col1:
        new_title = st.text_input("Title")
        new_author = st.text_input("Author")
        new_year = st.text_input("Year")
    with col2:
        new_desc = st.text_area("Description")
    add_submit = st.form_submit_button("Add")
    if add_submit and new_title and new_author:
        st.session_state.books.append(
            {
                "title": new_title,
                "author": new_author,
                "year": new_year,
                "status": "Available",
                "description": new_desc,
            }
        )
        st.session_state.embeddings = embed_books()

# Search and Q&A
query_tab, qa_tab = st.tabs(["Search", "Ask Library"])

with query_tab:
    query = st.text_input("Search by title or author")
    order = list(range(len(st.session_state.books)))
    if query:
        q_emb = embedder.encode([query])
        sims = cosine_similarity(q_emb, st.session_state.embeddings)[0]
        order = np.argsort(sims)[::-1]

    for idx in order:
        book = st.session_state.books[idx]
        header = f"{book['title']} - {book['author']} ({book['status']})"
        with st.expander(header, expanded=False):
            st.markdown(f"**Year**: {book['year']}")
            st.markdown(f"**Description**: {book['description']}")
            col1, col2, col3 = st.columns(3)
            if col1.button("Check-in / Check-out", key=f"toggle_{idx}"):
                book["status"] = "Borrowed" if book["status"] == "Available" else "Available"
            if col2.button("Edit", key=f"edit_{idx}"):
                st.session_state.edit_idx = idx
            if col3.button("Delete", key=f"delete_{idx}"):
                st.session_state.books.pop(idx)
                st.session_state.embeddings = embed_books()
                st.experimental_rerun()

            # Recommendations
            st.markdown("**Similar Books**")
            cur_emb = st.session_state.embeddings[idx]
            sims_all = cosine_similarity([cur_emb], st.session_state.embeddings)[0]
            sim_idx = np.argsort(sims_all)[::-1][1:4]
            for j in sim_idx:
                sim_book = st.session_state.books[j]
                st.write(f"{sim_book['title']} by {sim_book['author']} (score {sims_all[j]:.2f})")

with qa_tab:
    question = st.text_input("Ask a question about the library")
    if question:
        # Build context from top matching books
        q_emb = embedder.encode([question])
        sims = cosine_similarity(q_emb, st.session_state.embeddings)[0]
        top_k = np.argsort(sims)[::-1][:3]
        context = " ".join(
            [st.session_state.books[i]["description"] for i in top_k]
        )
        answer = qa_model(question=question, context=context)
        st.subheader("Answer")
        st.write(answer["answer"])

# Edit book modal
if "edit_idx" in st.session_state:
    i = st.session_state.edit_idx
    b = st.session_state.books[i]
    with st.form("edit_form"):
        st.subheader("Edit Book")
        et = st.text_input("Title", b["title"])
        ea = st.text_input("Author", b["author"])
        ey = st.text_input("Year", b["year"])
        ed = st.text_area("Description", b["description"])
        es = st.form_submit_button("Update")
        if es:
            b["title"], b["author"], b["year"], b["description"] = et, ea, ey, ed
            st.session_state.embeddings = embed_books()
            del st.session_state.edit_idx
            st.experimental_rerun()
