"""Vector embedding generator interface and local caching for AURA's RAG pipeline.

Transforms text strings into numerical dense vector embeddings for semantic
comparison and retrieval. Includes a zero-dependency deterministic local embedding
engine and an on-disk SHA-256 caching layer to prevent redundant recomputation.

Why are Embeddings Needed?
--------------------------
Computers cannot directly compute mathematical distances or conceptual similarity on raw
character strings. An embedding model projects discrete words, sentences, or paragraphs into
a continuous geometric vector space (e.g., R^256 or R^384). In this space:
- Semantically related phrases (e.g., "supervised learning" and "labeled training data")
  point in nearly the same directional orientation.
- Geometric metrics (such as the angle or cosine between vectors) allow us to search millions
  of passages in milliseconds based on semantic meaning rather than exact keyword matches.
"""

import os
import re
import math
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Base Embedding Interface
# ==============================================================================

class BaseEmbeddingModel(ABC):
    """Abstract interface for dense vector embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the generated vectors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single text string into an embedding vector.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the dense vector.
        """
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert a batch of text strings into embedding vectors.

        Args:
            texts: List of input text strings.

        Returns:
            List of embedding vectors.
        """
        return [self.embed_text(t) for t in texts]


# ==============================================================================
# 2. Local Deterministic Embedding Model (Educational & Zero-Dependency)
# ==============================================================================

class LocalTFIDFEmbeddingModel(BaseEmbeddingModel):
    """Zero-dependency, deterministic local embedding engine.

    Uses word-level and subword character n-gram feature hashing (the hashing trick)
    with sublinear term frequency weighting and L2 unit-norm projection:
    1. Tokenization: Deconstructs text into lowercased word tokens and character 3/4-grams.
       Subwords allow the model to recognize related morphological forms (e.g. 'learn', 'learning').
    2. Feature Hashing: Projects arbitrary vocabulary into a fixed-size vector space of dimension D
       using dual hashing (index bucket + unbiased sign flipping).
    3. Weighting & Normalization: Applies logarithmic term frequency weighting (1 + ln(tf))
       followed by L2 normalization, ensuring cosine similarity equals dot product.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Transform text into an L2-normalized dense vector of dimension D."""
        clean = (text or "").strip().lower()
        if not clean:
            # Return zero vector if input is empty
            return [0.0] * self._dimension

        vector = [0.0] * self._dimension

        # Extract words
        words = re.findall(r"\b[a-z0-9_-]+\b", clean)
        if not words:
            return [0.0] * self._dimension

        # Count frequencies of words and char n-grams
        term_counts: Dict[str, int] = {}
        for word in words:
            term_counts[word] = term_counts.get(word, 0) + 2  # words receive higher weight
            # Character n-grams for morphological overlap
            if len(word) >= 4:
                for n in (3, 4):
                    for i in range(len(word) - n + 1):
                        ngram = f"#{word[i:i+n]}"
                        term_counts[ngram] = term_counts.get(ngram, 0) + 1

        # Project into fixed-dimensional buckets using signed hashing
        for term, count in term_counts.items():
            # Logarithmic term frequency
            weight = 1.0 + math.log(count)

            # Primary hash for bucket index
            h1 = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16)
            bucket = h1 % self._dimension

            # Secondary hash for sign (+1 or -1) to reduce collision bias
            h2 = int(hashlib.sha1(term.encode("utf-8")).hexdigest(), 16)
            sign = 1.0 if (h2 % 2 == 0) else -1.0

            vector[bucket] += (sign * weight)

        # Apply L2 unit normalization (vector / ||vector||)
        norm_sq = sum(v * v for v in vector)
        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            vector = [round(v / norm, 6) for v in vector]

        return vector


# ==============================================================================
# 3. Embedding Caching Layer
# ==============================================================================

class EmbeddingCache:
    """Persistent on-disk cache for generated embedding vectors.

    Avoids recomputing embeddings for documents or chunks whose text has not changed.
    Embeddings are indexed by SHA-256 hash of the normalized text content.
    """

    def __init__(self, cache_path: Union[str, Path] = "data/embeddings_cache.json") -> None:
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, List[float]] = {}
        self.load()

    def _hash_key(self, text: str, model_id: str) -> str:
        """Generate a deterministic cache key from the text content and model identifier."""
        normalized = text.strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{model_id}:{digest}"

    def get(self, text: str, model_id: str) -> Optional[List[float]]:
        """Retrieve cached embedding vector if available."""
        key = self._hash_key(text, model_id)
        return self._cache.get(key)

    def set(self, text: str, model_id: str, vector: List[float]) -> None:
        """Store an embedding vector in the cache."""
        key = self._hash_key(text, model_id)
        self._cache[key] = list(vector)

    def contains(self, text: str, model_id: str) -> bool:
        """Check whether an embedding is already cached."""
        key = self._hash_key(text, model_id)
        return key in self._cache

    def save(self) -> None:
        """Persist current cache to JSON file on disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} cached embedding(s) to {self.cache_path}")
        except Exception as exc:
            logger.error(f"Failed to persist embedding cache to {self.cache_path}: {exc}")

    def load(self) -> None:
        """Load cached embeddings from disk."""
        if self.cache_path.exists() and self.cache_path.is_file():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached embedding(s) from {self.cache_path}")
            except Exception as exc:
                logger.warning(f"Could not read embedding cache from {self.cache_path}: {exc}")
                self._cache = {}
        else:
            self._cache = {}

    def clear(self) -> None:
        """Clear all in-memory and on-disk cached embeddings."""
        self._cache.clear()
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._cache)


# ==============================================================================
# 4. Cached Model Wrapper
# ==============================================================================

class CachedEmbeddingModel(BaseEmbeddingModel):
    """Wraps any BaseEmbeddingModel with a persistent EmbeddingCache.

    Automatically checks the cache before invoking the underlying model,
    saving significant computation for repeat documents and indexing passes.
    """

    def __init__(
        self,
        base_model: Optional[BaseEmbeddingModel] = None,
        cache: Optional[EmbeddingCache] = None,
        model_id: str = "local-tfidf-256",
    ) -> None:
        self.base_model = base_model or LocalTFIDFEmbeddingModel()
        self.cache = cache or EmbeddingCache()
        self.model_id = model_id
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def dimension(self) -> int:
        return self.base_model.dimension

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache hit and miss statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_cached": len(self.cache),
        }

    def embed_text(self, text: str) -> List[float]:
        """Return cached embedding if present, else compute, cache, and return."""
        clean = (text or "").strip()
        if not clean:
            return [0.0] * self.dimension

        cached_vec = self.cache.get(clean, self.model_id)
        if cached_vec is not None:
            self._cache_hits += 1
            return cached_vec

        self._cache_misses += 1
        vec = self.base_model.embed_text(clean)
        self.cache.set(clean, self.model_id, vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, utilizing the cache and saving updates."""
        results: List[List[float]] = []
        new_entries = False

        for t in texts:
            clean = (t or "").strip()
            cached_vec = self.cache.get(clean, self.model_id)
            if cached_vec is not None:
                self._cache_hits += 1
                results.append(cached_vec)
            else:
                self._cache_misses += 1
                vec = self.base_model.embed_text(clean)
                self.cache.set(clean, self.model_id, vec)
                results.append(vec)
                new_entries = True

        if new_entries:
            self.cache.save()

        return results


# ==============================================================================
# 5. Factory Function
# ==============================================================================

def get_embedding_model(
    model_type: Optional[str] = None,
    dimension: int = 256,
    use_cache: bool = True,
    cache_path: Union[str, Path] = "data/embeddings_cache.json",
) -> BaseEmbeddingModel:
    """Instantiate and return the configured embedding model.

    Args:
        model_type: Model identifier (defaults to 'local' for offline deterministic hashing).
        dimension: Dimensionality of vector representations (for local model).
        use_cache: Whether to wrap the model in an on-disk cache layer.
        cache_path: Filepath for the JSON embedding cache.

    Returns:
        Instance conforming to BaseEmbeddingModel.
    """
    resolved = (model_type or os.getenv("RAG_EMBEDDING_MODEL", "local")).lower().strip()

    if resolved in ("local", "tfidf", "default"):
        base_model = LocalTFIDFEmbeddingModel(dimension=dimension)
        model_id = f"local-tfidf-{dimension}"
    else:
        logger.warning(f"Unrecognized embedding model '{resolved}'. Falling back to 'local'.")
        base_model = LocalTFIDFEmbeddingModel(dimension=dimension)
        model_id = f"local-tfidf-{dimension}"

    if use_cache:
        cache = EmbeddingCache(cache_path=cache_path)
        return CachedEmbeddingModel(base_model=base_model, cache=cache, model_id=model_id)

    return base_model

