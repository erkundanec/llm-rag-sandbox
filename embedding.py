"""
Embedding utility for converting text to vectors.
Uses OpenAI's text-embedding-3-small model via OpenRouter.
"""

import requests
import os
import numpy as np

# Reuse your API key setup
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("Warning: OPENROUTER_API_KEY not set. Embeddings will fail.")

BASE_URL = "https://openrouter.ai/api/v1"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def get_embedding(text, model="openai/text-embedding-3-small"):
    """
    Convert text to an embedding vector.
    
    Args:
        text: String to embed
        model: Embedding model to use
        
    Returns:
        numpy array: Vector representation of the text (1536 dimensions)
    """
    url = f"{BASE_URL}/embeddings"
    
    data = {
        "model": model,
        "input": text
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        # Extract embedding from response
        embedding = result["data"][0]["embedding"]
        return np.array(embedding)
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(1536)  # Fallback


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Returns a score from -1 (opposite) to 1 (identical).
    Higher scores mean more similar.
    
    Args:
        vec1, vec2: numpy arrays
        
    Returns:
        float: Similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return dot_product / norm_product


if __name__ == "__main__":
    # Quick test
    text1 = "The cat sat on the mat"
    text2 = "A feline rested on the rug"
    text3 = "Python is a programming language"
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)
    
    print(f"Embedding dimension: {len(emb1)}")
    print(f"\nSimilarity (cat/feline): {cosine_similarity(emb1, emb2):.3f}")
    print(f"Similarity (cat/python): {cosine_similarity(emb1, emb3):.3f}")