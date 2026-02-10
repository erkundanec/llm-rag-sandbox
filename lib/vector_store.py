"""
Simple in-memory vector store for RAG.
Stores text chunks with their embeddings and provides semantic search.
"""

import numpy as np
from .embedding import get_embedding, cosine_similarity
import pickle


class SimpleVectorStore:
    """A minimal vector database."""

    def __init__(self):
        self.chunks = []  # List of text strings
        self.embeddings = []  # List of embedding vectors
        self.metadata = []  # List of metadata dicts

    def add_text(self, text, metadata=None):
        """
        Add a text chunk to the store.

        Args:
            text: String to add
            metadata: Optional dict like {"source": "doc1.txt"}
        """
        print(f"Adding: {text[:50]}...")

        # Get embedding for this text
        embedding = get_embedding(text)

        # Store everything
        self.chunks.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def search(self, query, top_k=3):
        """
        Find the most relevant chunks for a query.

        Args:
            query: Search string
            top_k: Number of results to return

        Returns:
            List of dicts: [{'text': ..., 'score': ..., 'metadata': ...}, ...]
        """
        if not self.chunks:
            return []

        print(f"\nSearching for: '{query}'")

        # Convert query to embedding
        query_embedding = get_embedding(query)

        # Calculate similarity with all chunks
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append(
                {"text": self.chunks[i], "score": score, "metadata": self.metadata[i]}
            )

        # Sort by score (highest first)
        similarities.sort(key=lambda x: x["score"], reverse=True)

        # Return top k
        results = similarities[:top_k]

        print(f"Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. Score: {r['score']:.3f} - {r['text'][:60]}...")

        return results

    def save(self, filepath):
        """Save to disk."""
        data = {
            "chunks": self.chunks,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "metadata": self.metadata,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {filepath}")

    def load(self, filepath):
        """Load from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.embeddings = [np.array(emb) for emb in data["embeddings"]]
        self.metadata = data["metadata"]
        print(f"Loaded {len(self.chunks)} chunks from {filepath}")

    def __len__(self):
        return len(self.chunks)
