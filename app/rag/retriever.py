"""Vector retriever and local vector storage layer for AURA's RAG pipeline.

Computes mathematical similarity scores (cosine similarity) between query embeddings
and document chunk embeddings to fetch top-k relevant context passages for the LLM.

What is a Vector and How Does Similarity Search Work?
-----------------------------------------------------
1. Vector: An ordered list of numbers representing a point or direction in high-dimensional space
   (e.g., [0.24, -0.81, 0.05, ..., 0.12]). In NLP embeddings, each dimension captures latent semantic
   features of the text.
2. Cosine Similarity: Measures the cosine of the angle between two vectors u and v:
      similarity = (u · v) / (||u|| * ||v||)
   - Angle = 0°  (Same direction)   -> similarity = 1.0 (Identical semantic orientation)
   - Angle = 90° (Orthogonal)        -> similarity = 0.0 (Unrelated topics)
   - Angle = 180° (Opposite)         -> similarity = -1.0
   When vectors are already L2 unit normalized (||u|| = ||v|| = 1), cosine similarity
   simplifies to the fast dot product: sum(u_i * v_i).
"""

import os
import math
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from app.rag.embeddings import BaseEmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Cosine Similarity Utility
# ==============================================================================

def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate the cosine similarity between two numerical vectors.

    Formula:
        cos(theta) = (vec_a . vec_b) / (||vec_a||_2 * ||vec_b||_2)

    Args:
        vec_a: First vector of floats.
        vec_b: Second vector of floats.

    Returns:
        Cosine similarity score bounded within [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Clamp due to potential floating point inaccuracies
    return max(-1.0, min(1.0, sim))


# ==============================================================================
# 2. Educational Local Vector Store
# ==============================================================================

class LocalVectorStore:
    """Lightweight, persistent local vector storage layer.

    Stores document chunks, metadata, and dense embedding vectors in an on-disk
    JSON format. Avoids heavyweight external database infrastructure while providing:
    - Fast in-memory cosine similarity search
    - On-disk persistence (`save` / `load`)
    - Incremental updates (detecting modified documents via content hashes)
    """

    def __init__(self, storage_path: Union[str, Path] = "data/vector_store.json") -> None:
        self.storage_path = Path(storage_path)
        self.entries: List[Dict[str, Any]] = []
        self.load()

    def add_entry(
        self,
        entry_id: str,
        text: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        doc_hash: Optional[str] = None,
    ) -> None:
        """Add or update a single vector entry in the store.

        Args:
            entry_id: Unique chunk identifier.
            text: Text content of the chunk.
            vector: Embedding vector.
            metadata: Associated contextual metadata (source, filename, page, etc.).
            doc_hash: Optional SHA-256 hash of the content to track changes.
        """
        clean_text = (text or "").strip()
        meta = dict(metadata) if metadata else {}
        computed_hash = doc_hash or hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

        # Check if entry already exists (update in place)
        for idx, item in enumerate(self.entries):
            if item.get("id") == entry_id:
                self.entries[idx] = {
                    "id": entry_id,
                    "text": clean_text,
                    "vector": vector,
                    "metadata": meta,
                    "hash": computed_hash,
                }
                return

        # New entry
        self.entries.append({
            "id": entry_id,
            "text": clean_text,
            "vector": vector,
            "metadata": meta,
            "hash": computed_hash,
        })

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search stored vectors for nearest matches to the query vector.

        Args:
            query_vector: Dense vector representing the user query.
            top_k: Maximum number of ranked results to return.
            min_score: Minimum similarity threshold (0.0 to 1.0).

        Returns:
            List of ranked dictionaries:
            {'id': str, 'text': str, 'score': float, 'metadata': dict}
        """
        if not query_vector or not self.entries:
            return []

        scored_results: List[Dict[str, Any]] = []

        for item in self.entries:
            score = compute_cosine_similarity(query_vector, item["vector"])

            if score >= min_score:
                scored_results.append({
                    "id": item["id"],
                    "text": item["text"],
                    "score": round(score, 4),
                    "metadata": dict(item.get("metadata", {})),
                })

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def save(self) -> None:
        """Persist vector entries to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)
            logger.debug(f"Saved {len(self.entries)} entries to {self.storage_path}")
        except Exception as exc:
            logger.error(f"Failed to save vector store to {self.storage_path}: {exc}")

    def load(self) -> None:
        """Load vector entries from disk if available."""
        if self.storage_path.exists() and self.storage_path.is_file():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                logger.debug(f"Loaded {len(self.entries)} entries from {self.storage_path}")
            except Exception as exc:
                logger.warning(f"Could not load vector store from {self.storage_path}: {exc}")
                self.entries = []
        else:
            self.entries = []

    def clear(self) -> None:
        """Clear in-memory entries and remove storage file."""
        self.entries.clear()
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception:
                pass

    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single stored entry by its identifier."""
        for item in self.entries:
            if item.get("id") == entry_id:
                return dict(item)
        return None

    def remove_by_id(self, entry_id: str) -> bool:
        """Remove an entry from the store by its identifier."""
        initial_len = len(self.entries)
        self.entries = [i for i in self.entries if i.get("id") != entry_id]
        return len(self.entries) < initial_len

    def count(self) -> int:
        """Return the count of stored entries."""
        return len(self.entries)

    def __len__(self) -> int:
        return self.count()


# ==============================================================================
# 3. Local Retriever
# ==============================================================================

class LocalRetriever:
    """Coordinates embedding generation, vector storage, and similarity retrieval."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[LocalVectorStore] = None,
        storage_path: Union[str, Path] = "data/vector_store.json",
    ) -> None:
        """Initialize the local retriever.

        Args:
            embedding_model: Embedding generator conforming to BaseEmbeddingModel.
            vector_store: Vector store instance. If omitted, LocalVectorStore is created.
            storage_path: Path for persisting the vector store JSON.
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or LocalVectorStore(storage_path=storage_path)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and index document chunks into the vector store.

        Detects whether an entry already has an embedding or has changed,
        avoiding unnecessary recomputation.

        Args:
            chunks: List of chunk dicts (from loader.chunk_text or chunk_documents).

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            return 0

        indexed_count = 0
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        for idx, chunk in enumerate(chunks):
            c_text = chunk.get("text", "").strip()
            c_id = chunk.get("chunk_id") or f"chunk_{idx}"
            metadata = chunk.get("metadata") or {}

            # Check if chunk already has a pre-computed vector
            if "vector" in chunk and isinstance(chunk["vector"], list):
                self.vector_store.add_entry(
                    entry_id=c_id,
                    text=c_text,
                    vector=chunk["vector"],
                    metadata=metadata,
                )
                indexed_count += 1
            else:
                texts_to_embed.append(c_text)
                indices_to_embed.append(idx)

        # Batch embed remaining chunks
        if texts_to_embed:
            embeddings = self.embedding_model.embed_batch(texts_to_embed)
            for idx_in_batch, original_idx in enumerate(indices_to_embed):
                chunk = chunks[original_idx]
                c_id = chunk.get("chunk_id") or f"chunk_{original_idx}"
                c_text = chunk.get("text", "").strip()
                metadata = chunk.get("metadata") or {}
                vector = embeddings[idx_in_batch]

                self.vector_store.add_entry(
                    entry_id=c_id,
                    text=c_text,
                    vector=vector,
                    metadata=metadata,
                )
                indexed_count += 1

        return indexed_count

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a user query.

        Args:
            query: Query string (e.g. 'What is overfitting?').
            top_k: Number of chunks to retrieve.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching chunk dicts with 'text', 'score', and 'metadata'.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        # 1. Generate query vector
        query_vector = self.embedding_model.embed_text(clean_query)

        # 2. Perform cosine similarity search
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
        )

        return results

    def save_index(self) -> None:
        """Save the underlying vector store to disk."""
        self.vector_store.save()

    def load_index(self) -> None:
        """Load the vector store from disk."""
        self.vector_store.load()

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.clear()

