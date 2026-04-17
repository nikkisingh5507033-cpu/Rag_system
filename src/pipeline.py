"""
Module to connect all components of the RAG system.
"""

from loader import load_text
from splitter import split_text
from embedder import load_embedder, create_embeddings
from vector_store import create_vector_store
from retriever import retrieve_chunks
from generator import initialize_client, generate_answer


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates the entire system.
    """
    
    def __init__(self, data_path):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_path (str): Path to the text data file
        """
        self.data_path = data_path
        self.chunks = []
        self.embedder = None
        self.index = None
        self.client = None
        
    def build(self):
        """
        Build the RAG system by loading, processing, and indexing data.
        """
        print("Loading text file...")
        text = load_text(self.data_path)
        if text is None:
            return False
        
        print("Splitting text into chunks...")
        self.chunks = split_text(text, chunk_size=200, overlap=50)
        print(f"Created {len(self.chunks)} chunks")
        
        print("Loading embedder model...")
        self.embedder = load_embedder()
        
        print("Creating embeddings...")
        embeddings = create_embeddings(self.chunks, self.embedder)
        
        print("Building vector store...")
        self.index = create_vector_store(embeddings)
        
        print("Initializing OpenAI client...")
        self.client = initialize_client()
        
        print("RAG system ready!\n")
        return True
    
    def query(self, question):
        """
        Answer a user question using the RAG system.
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The generated answer
        """
        # Create embedding for the question
        question_embedding = self.embedder.encode(question)
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_chunks(
            question_embedding,
            self.index,
            self.chunks,
            k=3
        )
        
        # Generate answer using LLM
        answer = generate_answer(question, relevant_chunks, self.client)
        
        return answer