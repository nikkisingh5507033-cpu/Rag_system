from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, vector_store):
    query_embedding = model.encode([query])[0]
    results = vector_store.search(query_embedding, k=2)
    return results
