"""
Demo script showing RAG in action.
"""

from build_index import build_index
from rag_system import RAGSystem
import os


def main():
    print("\n" + "=" * 70)
    print("RAG TUTORIAL - INTERACTIVE DEMO")
    print("=" * 70)
    
    # Build index if needed
    if not os.path.exists("rag_store.pkl"):
        print("\n📚 Building vector store (first time only)...")
        build_index()
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Example queries
    examples = [
        "What are embeddings?",
        "How does semantic search work?",
        "What is the purpose of chunking?",
    ]
    
    print("\n🎯 Running example comparisons...\n")
    
    for question in examples:
        rag.compare(question)
        input("\nPress Enter for next example...")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Now try your own questions!")
    print("=" * 70)
    print("Commands:")
    print("  - Ask any question about embeddings, RAG, vector DBs, etc.")
    print("  - Type 'compare: <question>' to see with/without RAG")
    print("  - Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('compare:'):
            rag.compare(user_input[8:].strip())
        else:
            answer = rag.query(user_input)
            print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
```

### Step 7: Dependencies

Create **requirements.txt**:
```
requests>=2.31.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

---

## 4. Complete Data Flow Explanation

Let me show you exactly how a query flows through the system:

### Example Query: "What are embeddings?"
```
┌─────────────────────────────────────────────────────────┐
│ USER INPUT: "What are embeddings?"                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: RETRIEVAL                                       │
│                                                         │
│ 1. Convert query to embedding                          │
│    "What are embeddings?"                              │
│    → [0.123, -0.456, 0.789, ..., 1536 numbers]        │
│                                                         │
│ 2. Compare with all stored chunks (cosine similarity)  │
│    doc1_embeddings.txt:     0.87 ✓✓✓                  │
│    doc4_semantic_search.txt: 0.65 ✓                    │
│    doc3_rag.txt:            0.52 ✓                     │
│    doc2_vector_db.txt:      0.48                       │
│    doc5_chunking.txt:       0.31                       │
│                                                         │
│ 3. Return top-3 chunks                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: AUGMENTATION                                    │
│                                                         │
│ Build enriched prompt:                                 │
│                                                         │
│ "Answer using ONLY the context below.                  │
│                                                         │
│  Context:                                              │
│  [Source 1: doc1_embeddings.txt]                       │
│  Embeddings are numerical representations of text...   │
│                                                         │
│  [Source 2: doc4_semantic_search.txt]                  │
│  Semantic search goes beyond keyword matching...       │
│                                                         │
│  [Source 3: doc3_rag.txt]                              │
│  Retrieval-Augmented Generation is a technique...      │
│                                                         │
│  Question: What are embeddings?                        │
│  Answer:"                                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: GENERATION (YOUR EXISTING CODE!)                │
│                                                         │
│ call_llm(augmented_prompt)                             │
│   ↓                                                     │
│ LLM reads context + generates answer                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│ "Embeddings are numerical representations of text that │
│  capture semantic meaning. Each piece of text is       │
│  converted into a vector with hundreds or thousands of │
│  dimensions. Similar texts have similar vectors,       │
│  measured by metrics like cosine similarity..."        │
└─────────────────────────────────────────────────────────┘
```

### Where RAG Improves Over Plain LLM

**Without RAG:**
```
User: "What are embeddings?"
LLM: "Embeddings are... [generic explanation from training data]"
     - Might be outdated
     - Can't reference your specific docs
     - Might hallucinate details
```

**With RAG:**
```
User: "What are embeddings?"
[System retrieves doc1_embeddings.txt]
LLM: "According to your documentation, embeddings are numerical 
      representations that capture semantic meaning..."
     - Grounded in YOUR documents
     - Can cite sources
     - More accurate and relevant