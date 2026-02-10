# In Python REPL or script
from lib.rag_system import RAGSystem

rag = RAGSystem()

# Ask questions about your knowledge base
rag.compare("What are embeddings?")
rag.compare("How does semantic search work?")
rag.compare("What is chunking in RAG?")