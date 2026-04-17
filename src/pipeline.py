from src.loader import load_data
from src.splitter import split_text
from src.embedder import get_embeddings
from src.vector_store import VectorStore
from src.retriever import retrieve
from src.generator import generate_answer

def build_pipeline(file_path):
    text = load_data(file_path)
    chunks = split_text(text)
    embeddings = get_embeddings(chunks)

    vector_store = VectorStore(len(embeddings[0]))
    vector_store.add(embeddings, chunks)

    return vector_store


def ask_question(query, vector_store):
    docs = retrieve(query, vector_store)
    context = "\n".join(docs)

    answer = generate_answer(query, context)
    return answer
