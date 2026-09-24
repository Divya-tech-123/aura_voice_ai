"""Comprehensive unit and integration tests for AURA Phase 6 - RAG.

Covers:
- Multi-format document loading (TXT, Markdown, PDF, DOCX)
- Unsupported files, missing files, and empty document directories
- Word-boundary text chunking with configurable size and overlap
- Deterministic local embedding generation, dimensions, and L2 normalization
- SHA-256 embedding caching and avoiding redundant recomputation
- Local vector store persistence, loading, and incremental updates
- Cosine similarity calculation, top-k retrieval, and similarity ranking
- LLM Brain RAG integration, context distinction, and source attribution
- Document indexing workflow CLI execution
"""

import os
import math
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.rag.loader import (
    Document,
    load_text_file,
    load_pdf_file,
    load_docx_file,
    load_document,
    load_directory,
    chunk_text,
    chunk_documents,
    index_documents,
    SUPPORTED_EXTENSIONS,
)
from app.rag.embeddings import (
    BaseEmbeddingModel,
    LocalTFIDFEmbeddingModel,
    EmbeddingCache,
    CachedEmbeddingModel,
    get_embedding_model,
)
from app.rag.retriever import (
    compute_cosine_similarity,
    LocalVectorStore,
    LocalRetriever,
)
from app.brain.prompts import format_rag_context, build_llm_messages
from app.brain.llm import LLMBrain, MockLLMProvider


# ==============================================================================
# 1. Document Loading Tests
# ==============================================================================

class TestDocumentLoader:
    """Tests for loading various document formats safely."""

    def test_load_text_file(self, tmp_path):
        """Verify reading plain text files with UTF-8 encoding."""
        test_file = tmp_path / "notes.txt"
        test_file.write_text("Machine learning is fascinating.", encoding="utf-8")

        content = load_text_file(test_file)
        assert content == "Machine learning is fascinating."

    def test_load_text_file_not_found(self):
        """Verify FileNotFoundError when target file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_text_file("nonexistent_file_xyz123.txt")

    def test_load_document_txt(self, tmp_path):
        """Verify load_document on .txt returns Document with metadata."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Sample content for RAG.", encoding="utf-8")

        docs = load_document(test_file)
        assert len(docs) == 1
        assert docs[0].text == "Sample content for RAG."
        assert docs[0].filename == "sample.txt"
        assert docs[0].metadata["file_type"] == ".txt"
        assert docs[0].page == 1

    def test_load_document_markdown(self, tmp_path):
        test_file = tmp_path / "guide.md"
        test_file.write_text("# Title\nMarkdown details.", encoding="utf-8")

        docs = load_document(test_file)
        assert len(docs) == 1
        assert "# Title" in docs[0].text
        assert docs[0].metadata["file_type"] == ".md"

    @patch("pypdf.PdfReader")
    def test_load_pdf_file_mocked(self, mock_pdf_reader, tmp_path):
        """Verify PDF loader extracts text page-by-page preserving page numbers."""
        test_pdf = tmp_path / "document.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy binary")

        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1: Introduction to AI."
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2: Neural Networks."

        mock_instance = MagicMock()
        mock_instance.pages = [page1, page2]
        mock_pdf_reader.return_value = mock_instance

        docs = load_pdf_file(test_pdf)
        assert len(docs) == 2
        assert docs[0].page == 1
        assert docs[0].text == "Page 1: Introduction to AI."
        assert docs[1].page == 2
        assert docs[1].text == "Page 2: Neural Networks."
        assert docs[0].metadata["file_type"] == ".pdf"

    @patch("docx.Document")
    def test_load_docx_file_mocked(self, mock_docx_doc, tmp_path):
        """Verify DOCX loader extracts paragraphs and tables."""
        test_docx = tmp_path / "document.docx"
        test_docx.write_bytes(b"dummy docx bytes")

        p1 = MagicMock()
        p1.text = "First paragraph in docx."
        p2 = MagicMock()
        p2.text = "Second paragraph."

        mock_instance = MagicMock()
        mock_instance.paragraphs = [p1, p2]
        mock_instance.tables = []
        mock_docx_doc.return_value = mock_instance

        docs = load_docx_file(test_docx)
        assert len(docs) == 1
        assert "First paragraph in docx." in docs[0].text
        assert "Second paragraph." in docs[0].text
        assert docs[0].metadata["file_type"] == ".docx"

    def test_unsupported_files_ignored(self, tmp_path):
        """Verify unsupported formats (.bin, .png, .exe) are safely ignored."""
        bin_file = tmp_path / "file.bin"
        bin_file.write_bytes(b"\x00\x01\x02")

        png_file = tmp_path / "image.png"
        png_file.write_bytes(b"fake png")

        assert load_document(bin_file) == []
        assert load_document(png_file) == []

    def test_empty_directory(self, tmp_path):
        """Verify scanning an empty directory safely returns empty list."""
        e = tmp_path / "empty_dir"
        e.mkdir()
        assert load_directory(e) == []

    def test_nonexistent_directory(self, tmp_path):
        """Verify scanning nonexistent directory returns empty list without crashing."""
        missing = tmp_path / "nonexistent"
        assert load_directory(missing) == []


# ==============================================================================
# 2. Text Chunking Tests
# ==============================================================================

class TestTextChunker:
    """Tests for splitting text into overlapping chunks."""

    def test_chunk_text_basic(self):
        """Verify basic text chunking preserves metadata and indices."""
        text = "Supervised learning trains models on labeled data to predict future outcomes accurately."
        meta = {"filename": "notes.txt", "page": 1}
        chunks = chunk_text(text, metadata=meta, chunk_size=40, chunk_overlap=10)

        assert len(chunks) > 1
        for idx, c in enumerate(chunks, 1):
            assert c["chunk_index"] == idx
            assert c["metadata"]["filename"] == "notes.txt"
            assert len(c["text"]) > 0

    def test_chunk_text_single_short_chunk(self):
        """Verify text smaller than chunk_size produces exactly 1 chunk."""
        text = "Short text example."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 1
        assert chunks[0]["total_chunks"] == 1
        assert chunks[0]["text"] == "Short text example."

    def test_chunk_text_empty(self):
        """Verify empty text produces no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_invalid_parameters(self):
        """Verify chunk_size <= 0 or overlap >= size raises ValueError."""
        with pytest.raises(ValueError):
            chunk_text("Test", chunk_size=0)

        with pytest.raises(ValueError):
            chunk_text("Test", chunk_size=50, chunk_overlap=50)

        with pytest.raises(ValueError):
            chunk_text("Test", chunk_size=50, chunk_overlap=60)

    def test_chunk_overlap_preserves_boundary_words(self):
        """Verify that words near split boundary appear in consecutive chunks."""
        text = "WordOne WordTwo WordThree WordFour WordFive WordSix WordSeven"
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(chunks) >= 2
        w1 = set(chunks[0]["text"].split())
        w2 = set(chunks[1]["text"].split())
        assert bool(w1.intersection(w2)), "Consecutive chunks should share overlapping words"

    def test_chunk_documents_batch(self):
        """Verify batch chunking across multiple Document instances."""
        doc1 = Document(text="Document 1 content.", metadata={"filename": "doc1.txt"})
        doc2 = Document(text="Document 2 content.", metadata={"filename": "doc2.txt"})

        all_chunks = chunk_documents([doc1, doc2], chunk_size=500, chunk_overlap=50)
        assert len(all_chunks) == 2
        assert all_chunks[0]["metadata"]["filename"] == "doc1.txt"
        assert all_chunks[1]["metadata"]["filename"] == "doc2.txt"


# ==============================================================================
# 3. Embedding Generation & Caching Tests
# ==============================================================================

class TestEmbeddings:
    """Tests for vector embeddings and caching."""

    def test_local_embedding_dimension(self):
        """Verify embedding model produces exact vector dimensionality."""
        model = LocalTFIDFEmbeddingModel(dimension=128)
        vec = model.embed_text("Machine learning")
        assert len(vec) == 128
        assert model.dimension == 128

    def test_local_embedding_l2_normalization(self):
        """Verify embedding vector has unit L2 norm (||v|| = 1.0)."""
        model = LocalTFIDFEmbeddingModel(dimension=256)
        vec = model.embed_text("Supervised learning and overfitting")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_local_embedding_deterministic(self):
        """Verify same text produces identical embedding vectors."""
        model = LocalTFIDFEmbeddingModel(dimension=128)
        assert model.embed_text("AI Notes") == model.embed_text("AI Notes")

    def test_local_embedding_empty_text(self):
        """Verify empty text yields zero vector."""
        model = LocalTFIDFEmbeddingModel(dimension=64)
        assert model.embed_text("") == [0.0] * 64

    def test_embedding_cache_hit_and_miss(self, tmp_path):
        """Verify EmbeddingCache stores and retrieves vectors without recomputation."""
        c = EmbeddingCache(cache_path=tmp_path / "c.json")
        m = CachedEmbeddingModel(base_model=LocalTFIDFEmbeddingModel(dimension=64), cache=c)

        m.embed_text("unique text")
        assert m.stats["misses"] == 1
        assert m.stats["hits"] == 0

        m.embed_text("unique text")
        assert m.stats["hits"] == 1

    def test_embedding_cache_persistence(self, tmp_path):
        """Verify EmbeddingCache saves and restores from disk."""
        c_path = tmp_path / "cache.json"
        c = EmbeddingCache(cache_path=c_path)
        c.set("sample", "m1", [0.1, 0.2, 0.3])
        c.save()

        c2 = EmbeddingCache(cache_path=c_path)
        assert c2.contains("sample", "m1")
        assert c2.get("sample", "m1") == [0.1, 0.2, 0.3]


# ==============================================================================
# 4. Vector Store & Cosine Similarity Tests
# ==============================================================================

class TestVectorStore:
    """Tests for local vector storage and cosine similarity calculation."""

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors should equal 1.0."""
        assert abs(compute_cosine_similarity([0.6, 0.8], [0.6, 0.8]) - 1.0) < 1e-4

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should equal 0.0."""
        assert abs(compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-4

    def test_cosine_similarity_empty_or_mismatched(self):
        """Mismatched vector lengths or empty vectors return 0.0 safely."""
        assert compute_cosine_similarity([], [1.0, 2.0]) == 0.0
        assert compute_cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_vector_store_persistence(self, tmp_path):
        """Verify LocalVectorStore saves to disk and restores accurately."""
        p = tmp_path / "store.json"
        s = LocalVectorStore(storage_path=p)
        s.add_entry("c1", "Supervised learning", [0.1, 0.2], metadata={"page": 1})
        s.save()

        s2 = LocalVectorStore(storage_path=p)
        assert len(s2) == 1
        assert s2.get_by_id("c1")["text"] == "Supervised learning"

    def test_vector_store_update_existing(self, tmp_path):
        """Verify updating existing id modifies entry without duplicates."""
        s = LocalVectorStore(storage_path=tmp_path / "s.json")
        s.add_entry("c1", "Old text", [0.1, 0.2])
        assert len(s) == 1

        s.add_entry("c1", "New text", [0.3, 0.4])
        assert len(s) == 1
        assert s.get_by_id("c1")["text"] == "New text"


# ==============================================================================
# 5. Local Retriever & Ranking Tests
# ==============================================================================

class TestLocalRetriever:
    """Tests for LocalRetriever query similarity and ranking."""

    def test_retriever_ranking(self, tmp_path):
        """Verify retriever ranks relevant chunks above unrelated chunks."""
        s = LocalVectorStore(storage_path=tmp_path / "s.json")
        r = LocalRetriever(vector_store=s)
        r.add_documents([
            {"chunk_id": "c1", "text": "Supervised learning uses labeled data", "metadata": {"filename": "ml.txt"}},
            {"chunk_id": "c2", "text": "The weather in London is rainy and cold", "metadata": {"filename": "w.txt"}},
        ])

        res = r.retrieve("What is supervised learning with labels?", top_k=1)
        assert len(res) == 1
        assert res[0]["id"] == "c1"
        assert res[0]["score"] > 0.0

    def test_retriever_top_k(self, tmp_path):
        """Verify top_k parameter restricts number of returned chunks."""
        s = LocalVectorStore(storage_path=tmp_path / "s.json")
        r = LocalRetriever(vector_store=s)
        r.add_documents([
            {"chunk_id": f"c_{i}", "text": f"Text passage number {i}", "metadata": {}}
            for i in range(5)
        ])

        results = r.retrieve("Text passage", top_k=2)
        assert len(results) <= 2

    def test_retriever_empty_query(self, tmp_path):
        """Verify empty query returns empty list."""
        r = LocalRetriever(vector_store=LocalVectorStore(storage_path=tmp_path / "s.json"))
        assert r.retrieve("") == []
        assert r.retrieve("   ") == []


# ==============================================================================
# 6. LLM Brain & RAG Integration Tests
# ==============================================================================

class TestLLMBrainRAGIntegration:
    """Tests for prompt formatting, grounding, and source attribution in LLMBrain."""

    def test_format_rag_context(self):
        """Verify format_rag_context generates proper document headers and source rules."""
        chunks = [{"id": "a1", "text": "Sample excerpt", "score": 0.8, "metadata": {"filename": "doc.txt", "page": 1}}]
        fmt = format_rag_context(chunks)
        assert "[Retrieved Context from Documents]" in fmt
        assert "doc.txt" in fmt
        assert "page 1" in fmt
        assert "Source:" in fmt

    def test_mock_llm_provider_rag(self):
        """Verify MockLLMProvider produces grounded response with cited sources when RAG context is present."""
        p = MockLLMProvider()
        chunks = [{"id": "c1", "text": "Supervised learning uses labeled data.", "score": 0.9, "metadata": {"filename": "notes.txt", "page": 1}}]
        msgs = build_llm_messages("What is supervised learning?", rag_chunks=chunks)
        resp = p.generate(msgs)

        assert "According to the reference document" in resp
        assert "Source:" in resp
        assert "notes.txt" in resp

    def test_brain_ask_with_rag(self, tmp_path):
        """Verify LLMBrain.ask_with_rag returns answer, chunks, and structured sources."""
        s = LocalVectorStore(storage_path=tmp_path / "store.json")
        r = LocalRetriever(vector_store=s)
        r.add_documents([
            {
                "chunk_id": "ch1",
                "text": "Overfitting happens when a model memorizes training noise.",
                "metadata": {"filename": "AI_Notes.txt", "page": 3},
            }
        ])

        b = LLMBrain(retriever=r)
        res = b.ask_with_rag("What is overfitting?")
        assert "answer" in res
        assert "chunks" in res
        assert "sources" in res
        assert len(res["sources"]) >= 1
        assert res["sources"][0]["page"] == 3
        assert "According to the reference document" in res["answer"]

    def test_brain_rag_disabled(self):
        """Verify RAG retrieval is bypassed when enable_rag is False."""
        mock_retriever = MagicMock()
        b = LLMBrain(retriever=mock_retriever, enable_rag=False)
        b.ask("What is machine learning?")
        assert not mock_retriever.retrieve.called


# ==============================================================================
# 7. Document Indexing Workflow CLI Tests
# ==============================================================================

class TestIndexingWorkflow:
    """Tests for the python -m app.rag.loader indexing workflow."""

    def test_index_documents_workflow(self, tmp_path):
        """Verify index_documents creates chunks and builds index from folder."""
        doc_dir = tmp_path / "my_docs"
        doc_dir.mkdir()

        (doc_dir / "doc1.txt").write_text("First document about machine learning.", encoding="utf-8")
        stats = index_documents(documents_dir=doc_dir)
        assert stats["documents_loaded"] == 1
        assert stats["chunks_indexed"] >= 1
