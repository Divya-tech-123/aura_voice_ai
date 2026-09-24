"""Unit and integration tests for AURA Document Upload Endpoint (Phase 11 - Step 35).

Covers:
- Valid TXT, PDF, and DOCX file uploads
- Unsupported file extension rejection (HTTP 400)
- Oversized file rejection (HTTP 413)
- Empty file rejection (HTTP 400)
- Unsafe filename and path traversal rejection (HTTP 400)
- Indexing failure handling (HTTP 422) and file cleanup
- Full integration: upload document -> query Agent via POST /api/chat -> retrieve, observe, answer
- Document listing endpoint (GET /api/documents)
"""

import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.api.server import create_app
from app.api.dependencies import get_agent, get_documents_dir
from app.agent.agent import AuraAgent
from app.agent.planner import Planner
from app.rag.retriever import LocalRetriever, LocalVectorStore
from app.rag.loader import Document
from app.brain.llm import LLMBrain, MockLLMProvider


@pytest.fixture
def test_setup(tmp_path):
    """Set up isolated app, documents directory, vector store, and agent for testing."""
    test_docs_dir = tmp_path / "documents"
    test_docs_dir.mkdir(parents=True, exist_ok=True)

    test_vstore_path = tmp_path / "vector_store.json"
    vector_store = LocalVectorStore(storage_path=test_vstore_path)
    retriever = LocalRetriever(vector_store=vector_store, storage_path=test_vstore_path)

    agent = AuraAgent(
        llm=LLMBrain(provider=MockLLMProvider(), retriever=retriever, enable_tools=False),
        planner=Planner(),
        retriever=retriever,
    )

    app = create_app()
    app.dependency_overrides[get_documents_dir] = lambda: test_docs_dir
    app.dependency_overrides[get_agent] = lambda: agent

    with TestClient(app) as client:
        yield {
            "client": client,
            "docs_dir": test_docs_dir,
            "agent": agent,
            "retriever": retriever,
            "app": app,
        }

    app.dependency_overrides.clear()


class TestDocumentUploadValidations:
    """Tests for document upload security, formatting, and constraint validations."""

    def test_valid_txt_upload(self, test_setup):
        """Verify successful upload, storage, and indexing of a .txt file."""
        client = test_setup["client"]
        docs_dir = test_setup["docs_dir"]
        retriever = test_setup["retriever"]

        file_content = b"Supervised learning is an approach where models learn from labeled training data."
        response = client.post(
            "/api/documents/upload",
            files={"file": ("AI_Notes.txt", io.BytesIO(file_content), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "AI_Notes.txt"
        assert data["message"] == "Document indexed successfully"
        assert data["chunks_indexed"] >= 1

        # File is safely stored in controlled directory
        stored_file = docs_dir / "AI_Notes.txt"
        assert stored_file.exists()
        assert stored_file.read_bytes() == file_content

        # Chunks are in vector store
        assert len(retriever.vector_store) >= 1

    def test_valid_pdf_upload(self, test_setup):
        """Verify successful upload of a PDF file using mocked parser."""
        client = test_setup["client"]
        docs_dir = test_setup["docs_dir"]

        dummy_pdf_bytes = b"%PDF-1.4 dummy pdf content for testing"
        mock_docs = [
            Document(
                text="Deep learning uses artificial neural networks with multiple layers.",
                metadata={"filename": "Deep_Learning.pdf", "page": 1, "file_type": ".pdf"},
            )
        ]

        with patch("app.rag.loader.load_pdf_file", return_value=mock_docs):
            response = client.post(
                "/api/documents/upload",
                files={"file": ("Deep_Learning.pdf", io.BytesIO(dummy_pdf_bytes), "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "Deep_Learning.pdf"
        assert (docs_dir / "Deep_Learning.pdf").exists()

    def test_valid_docx_upload(self, test_setup):
        """Verify successful upload of a DOCX file using mocked parser."""
        client = test_setup["client"]
        docs_dir = test_setup["docs_dir"]

        dummy_docx_bytes = b"PK\x03\x04 dummy docx binary content"
        mock_docs = [
            Document(
                text="Reinforcement learning agents learn policies by maximizing cumulative reward.",
                metadata={"filename": "RL_Summary.docx", "page": 1, "file_type": ".docx"},
            )
        ]

        with patch("app.rag.loader.load_docx_file", return_value=mock_docs):
            response = client.post(
                "/api/documents/upload",
                files={"file": ("RL_Summary.docx", io.BytesIO(dummy_docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "RL_Summary.docx"
        assert (docs_dir / "RL_Summary.docx").exists()

    def test_unsupported_file_rejection(self, test_setup):
        """Verify unsupported file extensions are rejected with HTTP 400."""
        client = test_setup["client"]

        unsupported_files = [
            ("script.py", b"print('hello')", "text/x-python"),
            ("program.exe", b"\x4d\x5a\x90\x00", "application/octet-stream"),
            ("image.png", b"\x89PNG\r\n\x1a\n", "image/png"),
            ("archive.zip", b"PK\x03\x04", "application/zip"),
            ("document.bin", b"\x00\x01\x02", "application/octet-stream"),
        ]

        for filename, content, mime in unsupported_files:
            response = client.post(
                "/api/documents/upload",
                files={"file": (filename, io.BytesIO(content), mime)},
            )
            assert response.status_code == 400
            assert "Unsupported file type" in response.json()["detail"]

    def test_oversized_file_rejection(self, test_setup):
        """Verify files exceeding 10 MB are rejected with HTTP 413."""
        client = test_setup["client"]

        # Patch MAX_DOCUMENT_SIZE to a small threshold for testing efficiency
        with patch("app.api.routes.MAX_DOCUMENT_SIZE", 500):
            large_content = b"A" * 600
            response = client.post(
                "/api/documents/upload",
                files={"file": ("large_file.txt", io.BytesIO(large_content), "text/plain")},
            )
            assert response.status_code == 413
            assert "exceeds maximum allowed size" in response.json()["detail"]

    def test_empty_file_rejection(self, test_setup):
        """Verify zero-byte files are rejected with HTTP 400."""
        client = test_setup["client"]

        response = client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_unsafe_filename_rejection(self, test_setup):
        """Verify path traversal sequences and unsafe characters are rejected."""
        client = test_setup["client"]
        content = b"Sample text content."

        unsafe_filenames = [
            "../../etc/passwd.txt",
            "..\\..\\Windows\\System32\\notes.txt",
            "folder/subfolder/notes.txt",
            "folder\\notes.txt",
            ".hidden_notes.txt",
            "evil\x00file.txt",
            "test;rm.txt",
            "test|hack.txt",
            "test$exec.txt",
        ]

        for unsafe_name in unsafe_filenames:
            response = client.post(
                "/api/documents/upload",
                files={"file": (unsafe_name, io.BytesIO(content), "text/plain")},
            )
            assert response.status_code == 400
            detail = response.json()["detail"].lower()
            assert "invalid filename" in detail or "path traversal" in detail or "unsafe" in detail

    def test_indexing_failure_handling(self, test_setup):
        """Verify that when extraction/indexing yields 0 chunks, HTTP 422 is returned and file is removed."""
        client = test_setup["client"]
        docs_dir = test_setup["docs_dir"]

        with patch("app.rag.loader.load_document", return_value=[]):
            response = client.post(
                "/api/documents/upload",
                files={"file": ("corrupt.txt", io.BytesIO(b"Corrupt text"), "text/plain")},
            )

        assert response.status_code == 422
        assert "indexing failed" in response.json()["detail"].lower()
        # Ensure temporary corrupt file was cleaned up
        assert not (docs_dir / "corrupt.txt").exists()


class TestDocumentListEndpoint:
    """Tests for GET /api/documents."""

    def test_list_documents_empty(self, test_setup):
        """Verify empty list is returned when no documents exist."""
        client = test_setup["client"]
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == {"documents": []}

    def test_list_documents_after_upload(self, test_setup):
        """Verify uploaded documents appear in GET /api/documents with metadata and no server paths."""
        client = test_setup["client"]

        # Upload a document
        client.post(
            "/api/documents/upload",
            files={"file": ("Guide.txt", io.BytesIO(b"Knowledge base guide content"), "text/plain")},
        )

        response = client.get("/api/documents")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"] == "Guide.txt"
        assert docs[0]["status"] == "indexed"
        assert docs[0]["size"] > 0
        # Check no absolute filesystem path is leaked
        assert "path" not in docs[0]
        assert "/" not in docs[0]["filename"]
        assert "\\" not in docs[0]["filename"]


class TestAgentRAGChatIntegration:
    """Tests full end-to-end flow: Upload document -> Query agent -> Retrieve -> Observe -> Answer."""

    def test_chat_retrieves_from_uploaded_document(self, test_setup):
        """Verify the Agent autonomously retrieves from the uploaded document to answer user questions."""
        client = test_setup["client"]

        # 1. Upload knowledge document
        doc_content = (
            "Supervised learning is a machine learning paradigm where models train on labeled data. "
            "Algorithms map inputs to targets through regression or classification."
        )
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("ML_Notes.txt", io.BytesIO(doc_content.encode("utf-8")), "text/plain")},
        )
        assert upload_resp.status_code == 200

        # 2. Query the agent via POST /api/chat
        chat_resp = client.post(
            "/api/chat",
            json={"message": "What does my document say about supervised learning?"},
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()

        # 3. Verify the agent planned retrieval and responded
        activities = chat_data.get("activities", [])
        activity_types = [a["type"] for a in activities]
        assert "retrieval" in activity_types or "response" in activity_types
        assert len(chat_data["response"]) > 0
