"""
Complete RAG System: Retrieval + Augmentation + Generation
"""

import os
import sys
from .vector_store import SimpleVectorStore

# ============================================================================
# YOUR EXISTING API CLIENT CODE (REUSED AS-IS)
# ============================================================================

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY not set")
    sys.exit(1)

BASE_URL = "https://openrouter.ai/api/v1"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _extract_reply(result):
    try:
        choices = result.get("choices", [])
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    return msg.get("content")
                if first.get("text"):
                    return first.get("text")
        if isinstance(result.get("output"), str):
            return result.get("output")
    except:
        pass
    return ""


def call_llm(messages, model="gpt-3.5-turbo"):
    """Call LLM with messages. YOUR EXISTING CODE, UNCHANGED."""
    url = f"{BASE_URL}/chat/completions"
    data = {"model": model, "messages": messages}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return _extract_reply(result) or str(result)
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# RAG SYSTEM (NEW)
# ============================================================================


class RAGSystem:
    """Simple Retrieval-Augmented Generation system."""

    def __init__(self, vector_store_path="rag_store.pkl"):
        """Load the vector store."""
        print("Loading vector store...")
        self.store = SimpleVectorStore()
        self.store.load(vector_store_path)
        print(f"Ready with {len(self.store)} chunks\n")

    def query(self, question, top_k=3, use_rag=True):
        """
        Answer a question with or without RAG.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            use_rag: If False, skip retrieval (for comparison)

        Returns:
            str: The answer
        """
        if not use_rag:
            # Direct LLM call (baseline)
            print("🤖 Querying LLM directly (no RAG)...\n")
            messages = [{"role": "user", "content": question}]
            return call_llm(messages)

        # STEP 1: RETRIEVAL
        print("🔍 RETRIEVAL: Finding relevant chunks...")
        results = self.store.search(question, top_k=top_k)

        if not results:
            print("No relevant chunks found\n")
            messages = [{"role": "user", "content": question}]
            return call_llm(messages)

        # STEP 2: AUGMENTATION
        print("\n📝 AUGMENTATION: Building enriched prompt...")
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{result['text']}")

        context = "\n\n".join(context_parts)

        # Build the augmented prompt
        augmented_prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""

        print(f"Context length: {len(context)} characters\n")

        # STEP 3: GENERATION
        print("💬 GENERATION: Calling LLM with context...\n")
        messages = [{"role": "user", "content": augmented_prompt}]
        return call_llm(messages)

    def compare(self, question, top_k=3):
        """Compare LLM with and without RAG side-by-side."""
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        # Without RAG
        print("\n" + "-" * 70)
        print("WITHOUT RAG (LLM baseline):")
        print("-" * 70)
        without = self.query(question, use_rag=False)
        print(f"{without}\n")

        # With RAG
        print("-" * 70)
        print("WITH RAG (Retrieval + LLM):")
        print("-" * 70)
        with_rag = self.query(question, top_k=top_k, use_rag=True)
        print(f"{with_rag}\n")

        print("=" * 70)


if __name__ == "__main__":
    # Interactive mode
    rag = RAGSystem()

    print("RAG System Ready!")
    print("Commands:")
    print("  - Type your question to use RAG")
    print("  - Type 'compare: <question>' to see with/without RAG")
    print("  - Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        if user_input.lower().startswith("compare:"):
            question = user_input[8:].strip()
            rag.compare(question)
        else:
            answer = rag.query(user_input)
            print(f"\nAnswer: {answer}\n")
