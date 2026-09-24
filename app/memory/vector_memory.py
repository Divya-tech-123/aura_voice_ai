"""Long-term vector memory interface and local store for AURA.

Enables semantic retrieval of past user facts, preferences, and long-term knowledge.

Architectural Distinction:
--------------------------
1. Short-Term Memory (ConversationMemory):
   - Scope: The immediate, active conversation.
   - Organization: Strictly ordered, chronological buffer of dialogue turns.
   - Usage: Fed directly into the LLM context on every query turn.
   - Lifecycle: Cleared at session end or trimmed via sliding-window FIFO.

2. Long-Term / Vector Memory (VectorMemory):
   - Scope: Enduring facts, user preferences, and knowledge across past sessions.
   - Organization: Unordered collection of documents indexed for semantic similarity.
   - Usage: Queried on-demand when a prompt needs relevant background knowledge (RAG).
   - Lifecycle: Persists across sessions; retrieved only when semantically relevant.

Note for Phase 4:
No external vector databases or heavyweight embedding models are introduced here.
A clean abstraction with a lightweight, zero-dependency token-similarity local engine
is provided for testing and demonstration.
"""

from abc import ABC, abstractmethod
import time
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseVectorMemory(ABC):
    """Abstract interface defining long-term semantic memory storage and retrieval."""

    @abstractmethod
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a text document with optional metadata.

        Args:
            text: Text snippet to persist.
            metadata: Associated contextual metadata (e.g. category, source, timestamp).

        Returns:
            Unique identifier of the stored document.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search memory for snippets relevant to the query.

        Args:
            query: Query string to match against memory.
            top_k: Maximum number of relevant items to return.

        Returns:
            List of matching items sorted by relevance score, each containing:
            {'id': str, 'text': str, 'score': float, 'metadata': dict}
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored long-term memories."""
        pass


class VectorMemory(BaseVectorMemory):
    """Lightweight local in-memory semantic memory store for AURA.

    Uses word-token overlap and Jaccard similarity scoring to simulate semantic
    retrieval without external embedding dependencies.
    """

    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self._counter: int = 0

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory document with optional metadata.

        Args:
            text: Text snippet to persist.
            metadata: Associated contextual metadata.

        Returns:
            Unique document ID.
        """
        clean_text = str(text).strip()
        self._counter += 1
        doc_id = f"mem_{self._counter}"

        meta = dict(metadata) if metadata else {}
        meta.setdefault("timestamp", time.time())

        doc = {
            "id": doc_id,
            "text": clean_text,
            "metadata": meta,
        }
        self.documents.append(doc)
        logger.debug(f"Stored long-term memory document: {doc_id}")
        return doc_id

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search memory for snippets relevant to the query using token similarity.

        Args:
            query: Query string to match against memory.
            top_k: Number of relevant items to return.

        Returns:
            List of matching items with id, text, score, and metadata.
        """
        if not query or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_results: List[Dict[str, Any]] = []

        for doc in self.documents:
            doc_tokens = self._tokenize(doc["text"])
            score = self._calculate_similarity(query_tokens, doc_tokens)

            if score > 0.0:
                scored_results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": round(score, 4),
                    "metadata": dict(doc["metadata"]),
                })

        # Sort descending by relevance score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored documents."""
        return [dict(doc) for doc in self.documents]

    def clear(self) -> None:
        """Clear all stored documents from memory."""
        self.documents.clear()
        self._counter = 0

    def count(self) -> int:
        """Return the count of stored memory documents."""
        return len(self.documents)

    def __len__(self) -> int:
        return self.count()

    @staticmethod
    def _tokenize(text: str) -> set:
        """Convert string to a set of lowercased alphanumeric tokens."""
        return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

    @staticmethod
    def _calculate_similarity(query_tokens: set, doc_tokens: set) -> float:
        """Compute token overlap and root-prefix similarity score between query and document."""
        if not query_tokens or not doc_tokens:
            return 0.0

        matches = 0.0
        for q in query_tokens:
            if q in doc_tokens:
                matches += 1.0
            elif len(q) >= 4 and any(d.startswith(q[:4]) or q.startswith(d[:4]) for d in doc_tokens if len(d) >= 4):
                matches += 0.75  # Root prefix match (e.g. live/lives, work/working)

        union = len(query_tokens | doc_tokens)
        return (matches / union) if union > 0 else 0.0

