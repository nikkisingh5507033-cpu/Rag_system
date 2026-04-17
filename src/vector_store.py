import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings))
        self.chunks.extend(chunks)

    def search(self, query_embedding, k=2):
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        return [self.chunks[i] for i in indices[0]]
