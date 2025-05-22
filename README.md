
---

# Mini Library Management System with AI-Powered Search

This interactive library management system is built with **Streamlit** and incorporates **AI-powered semantic search** and **question-answering** features. The system enables users to manage books, track check-ins and check-outs, and ask questions about the library's contents using AI.

## Features

* **Add/Edit/Delete Books**: Manage your library by adding, editing, or removing books.
* **Check-in/Check-out**: Keep track of books' availability or borrowing status.
* **Semantic Search**: Search books by title, author, or description with advanced AI-powered semantic search.
* **Ask Questions**: Ask specific questions about the books with the question-answering model.
* **Book Recommendations**: Receive AI-generated book recommendations based on similarity.

## Tech Stack

* **Python 3.11**
* **Streamlit** for the web interface
* **Sentence-Transformers** for semantic search
* **HuggingFace Transformers** for Q\&A capabilities
* **Scikit-learn** for cosine similarity and vector handling
* **Torch** as the deep learning backend

## How to Run Locally

1. **Extract the contents of the zip file** to a folder on your machine.

2. **Create and activate a virtual environment** (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```
