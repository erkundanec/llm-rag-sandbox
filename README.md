# LLM RAG Sandbox

A Retrieval-Augmented Generation (RAG) system sandbox for experimenting with embedding-based document retrieval and semantic search.

## Project Overview

This project implements a RAG pipeline that:

- Embeds documents using vector embeddings
- Stores and retrieves vectors from a vector database
- Performs semantic search across knowledge documents
- Demonstrates chunking and document processing techniques

## Project Structure

```
llm-rag-sandbox/
├── README.md                      # This file
├── requirement.txt                # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── RAG_basics.ipynb              # Jupyter notebook with RAG fundamentals
│
├── embedding.py                  # Document embedding functions
├── vector_store.py               # Vector database operations
├── rag_system.py                 # Main RAG system implementation
├── build_index.py                # Index building script
├── demo.py                       # Demo/example usage
│
├── test_openrautoer.py           # Test suite
│
└── knowledge_base/               # Document embeddings & vectors
    ├── doc1_embeddings.txt       # Embedding data
    ├── doc2_vector_db.txt        # Vector database file
    ├── doc3_rag.txt              # RAG reference documents
    ├── doc4_semantic_search.txt  # Search examples
    └── doc5_chunking.txt         # Chunking examples
```

## Installation

### Prerequisites

- Python 3.12.10
- pip/venv

### Setup

1. **Create and activate virtual environment:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirement.txt
   ```

## Usage

### Build Index

```bash
python build_index.py
```

### Run Demo

```bash
python demo.py
```

### Run Tests

```bash
python -m pytest test_openrautoer.py
```

### Jupyter Notebook

```bash
jupyter notebook RAG_basics.ipynb
```

## Dependencies

- **requests** ≥2.31.0 — HTTP library for API calls
- **numpy** ≥1.24.0 — Numerical computing
- **python-dotenv** ≥1.0.0 — Environment variable management

## Module Descriptions

| Module             | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `embedding.py`     | Handles document-to-vector conversion and embedding operations |
| `vector_store.py`  | Manages vector storage, indexing, and similarity search        |
| `rag_system.py`    | Orchestrates the full RAG pipeline                             |
| `build_index.py`   | Processes documents and builds the search index                |
| `demo.py`          | Demonstrates RAG functionality with example queries            |
| `RAG_basics.ipynb` | Interactive notebook explaining RAG concepts                   |

## RAG Pipeline Overview

1. **Document Chunking** — Split documents into manageable chunks
2. **Embedding** — Convert text to vector embeddings
3. **Indexing** — Store vectors in a searchable database
4. **Query Processing** — Embed user query and search for similar documents
5. **Retrieval** — Return most relevant documents based on similarity

## Configuration

Create a `.env` file for sensitive configuration (API keys, model names, etc.):

```ini
OPENAI_API_KEY=your_key_here
MODEL_NAME=text-embedding-ada-002
```

## Notes

- Vector databases are stored in `knowledge_base/` directory
- Embeddings are cached to avoid redundant API calls
- See `RAG_basics.ipynb` for detailed walkthroughs

## License

MIT (or your preferred license)
