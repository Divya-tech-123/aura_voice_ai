"""Interactive demonstration of AURA Phase 6 — RAG.

Shows:
1. Document indexing & embedding generation
2. Question submission ("What is supervised learning?" and "What is overfitting?")
3. Retrieved chunk text snippet
4. Cosine similarity score
5. Final LLM answer
6. Explicit cited source metadata
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.rag.loader import index_documents
from app.rag.retriever import LocalRetriever
from app.brain.llm import LLMBrain


def run_rag_demonstration() -> None:
    print("=" * 65)
    print("          AURA Phase 6: RAG Pipeline Demonstration       ")
    print("=" * 65)

    # 1. Index local documents
    print("\n[Step 1] Scanning and indexing 'documents/' directory...")
    stats = index_documents("documents")
    print(f"Indexed {stats.get('chunks_indexed', 0)} chunk(s) from {stats.get('documents_loaded', 0)} document(s).")

    # 2. Initialize Retriever and LLMBrain
    retriever = LocalRetriever()
    brain = LLMBrain(retriever=retriever)

    demo_questions = [
        "What is supervised learning?",
        "What is overfitting?",
    ]

    for idx, question in enumerate(demo_questions, 1):
        print("\n" + "=" * 65)
        print(f"Demo Query #{idx}: '{question}'")
        print("=" * 65)

        # Retrieve top relevant chunk
        chunks = retriever.retrieve(question, top_k=1)
        if not chunks:
            print("No relevant chunks retrieved.")
            continue

        top = chunks[0]
        meta = top.get("metadata", {})
        score = top.get("score", 0.0)
        filename = meta.get("filename", "unknown")
        page = meta.get("page", 1)

        print("\n[1. Retrieved Chunk]")
        print(f"  Source File : {filename}")
        print(f"  Page / Part : Page {page}")
        snippet = top.get("text", "").strip().replace("\n", " ")[:200]
        print(f"  Text Excerpt:\n  \"{snippet}...\"")

        print("\n[2. Similarity Score]")
        print(f"  Cosine Similarity Score : {score:.4f}")

        # Ask LLM Brain
        print("\n[3. Final Grounded LLM Answer]")
        result = brain.ask_with_rag(question)
        print(f"{result['answer']}")

        print("\n[4. Cited Source Metadata]")
        for src in result.get("sources", []):
            print(f"  - File: {src.get('filename')} | Page: {src.get('page')} | Similarity: {src.get('score'):.4f}")

    print("\n" + "=" * 65)
    print("RAG Demonstration Complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_rag_demonstration()

