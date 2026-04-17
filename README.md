# Simple RAG System

This is a basic **RAG (Retrieval-Augmented Generation)** project built using Python.

## What is RAG?

RAG = Retrieve + Generate

* Retrieve relevant text from documents
* Generate answer using an AI model

---

## How it works

1. Load text file
2. Split into small chunks
3. Convert chunks into embeddings
4. Store in FAISS vector database
5. Retrieve relevant chunks
6. Generate answer using LLM

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Run

```bash
python app.py
```

---

## Example

**Question:**

```
What is RAG?
```

**Answer:**

```
RAG stands for Retrieval-Augmented Generation...
```

---

## Tech Used

* Python
* FAISS
* Sentence Transformers
* OpenAI API

---

## Note

This is a simple project for learning how RAG works.
