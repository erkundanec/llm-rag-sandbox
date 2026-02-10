"""
Build the vector store from knowledge base documents.
Run this once to create your searchable index.
"""

import os
from vector_store import SimpleVectorStore


def load_documents(directory):
    """Load all .txt files from a directory."""
    documents = []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                documents.append((filename, content))
    
    return documents


def build_index(kb_directory="knowledge_base", output_file="rag_store.pkl"):
    """
    Build and save a vector store.
    
    Args:
        kb_directory: Folder containing your documents
        output_file: Where to save the index
    """
    print("="* 60)
    print("BUILDING VECTOR STORE")
    print("="* 60)
    
    # Load documents
    print(f"\nLoading documents from {kb_directory}/...")
    documents = load_documents(kb_directory)
    print(f"Found {len(documents)} documents\n")
    
    # Create store
    store = SimpleVectorStore()
    
    # Add each document (simple: 1 doc = 1 chunk)
    for filename, content in documents:
        store.add_text(content, metadata={"source": filename})
    
    # Save
    print(f"\nSaving...")
    store.save(output_file)
    
    print("\n" + "="* 60)
    print(f"DONE! Created index with {len(store)} chunks")
    print("="* 60)
    
    return store


if __name__ == "__main__":
    store = build_index()
    
    # Quick test
    print("\n" + "="* 60)
    print("QUICK TEST")
    print("="* 60)
    results = store.search("What are embeddings?", top_k=2)