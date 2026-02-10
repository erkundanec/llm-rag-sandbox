"""
RAG Library - Core modules for Retrieval-Augmented Generation.
"""

from .embedding import get_embedding, cosine_similarity
from .vector_store import SimpleVectorStore
from .rag_system import RAGSystem

__all__ = [
    "get_embedding",
    "cosine_similarity",
    "SimpleVectorStore",
    "RAGSystem",
]
