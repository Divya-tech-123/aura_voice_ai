"""Document loader and text chunker for AURA's RAG pipeline.

Loads text, markdown, PDF, and DOCX documents from disk safely, extracts metadata,
and splits text into semantically cohesive chunks with configurable overlap.

Why is Chunking Essential?
--------------------------
1. Embedding Fidelity: Embedding models compress an entire passage into a single fixed-size
   vector (e.g. 256 or 384 dimensions). Embedding a 50-page document as one vector washes out
   specific details. Splitting into chunks preserves fine-grained semantic focus.
2. LLM Context Window: Feeding entire books or manuals into an LLM context is costly, slow,
   and risks exceeding token limits or causing "lost in the middle" attention degradation.

Why is Chunk Overlap Essential?
-------------------------------
When text is sliced at fixed boundaries, ideas spanning consecutive sentences can be bisected.
For example, if Chunk 1 ends with "Overfitting happens when a model learns noise," and Chunk 2
begins with "It can be prevented using dropout and regularization," a query asking "How to prevent
overfitting?" might fail to match either chunk strongly. Overlap ensures that boundary sentences
appear in both chunks, maintaining contextual continuity.
"""

import os
import re
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Extensions supported by the loader
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class Document:
    """Represents an extracted document passage with contextual metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Return the source file path string."""
        return str(self.metadata.get("source", "unknown"))

    @property
    def filename(self) -> str:
        """Return the base filename."""
        return str(self.metadata.get("filename", "unknown"))

    @property
    def page(self) -> Optional[int]:
        """Return the page number if available (e.g. from PDFs)."""
        return self.metadata.get("page")


def load_text_file(file_path: Union[str, Path]) -> str:
    """Read plain text or markdown content safely from disk.

    Args:
        file_path: Path to the text or markdown file.

    Returns:
        String content of the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    # Try UTF-8 first, with fallback to latin-1 to handle diverse encodings safely
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path.name}; falling back to latin-1.")
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            return f.read()


def load_pdf_file(file_path: Union[str, Path]) -> List[Document]:
    """Read a PDF document page-by-page using pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document instances, one per non-empty page, preserving page metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        import pypdf
    except ImportError:
        logger.error("pypdf is not installed. Run 'pip install pypdf' to parse PDF files.")
        return []

    documents: List[Document] = []
    try:
        reader = pypdf.PdfReader(str(path))
        num_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                page_text = page.extract_text() or ""
            except Exception as extract_err:
                logger.warning(f"Failed to extract text from page {page_num} of {path.name}: {extract_err}")
                page_text = ""

            clean_text = page_text.strip()
            if clean_text:
                documents.append(
                    Document(
                        text=clean_text,
                        metadata={
                            "source": str(path.resolve()),
                            "filename": path.name,
                            "file_type": ".pdf",
                            "page": page_num,
                            "total_pages": num_pages,
                            "char_count": len(clean_text),
                        },
                    )
                )

        logger.debug(f"Extracted {len(documents)} page(s) from PDF {path.name}")
        return documents

    except Exception as exc:
        logger.error(f"Error reading PDF file {path}: {exc}")
        return []


def load_docx_file(file_path: Union[str, Path]) -> List[Document]:
    """Read a Microsoft Word (.docx) document using python-docx.

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of Document instances representing paragraphs/sections.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    try:
        import docx
    except ImportError:
        logger.error("python-docx is not installed. Run 'pip install python-docx' to parse DOCX files.")
        return []

    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n\n".join(paragraphs).strip()
        if not full_text:
            return []

        return [
            Document(
                text=full_text,
                metadata={
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "file_type": ".docx",
                    "page": 1,
                    "char_count": len(full_text),
                },
            )
        ]
    except Exception as exc:
        logger.error(f"Error reading DOCX file {path}: {exc}")
        return []


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """Read a single document safely based on its extension.

    Supports:
    - .txt  (Plain text)
    - .md   (Markdown)
    - .pdf  (Portable Document Format)
    - .docx (Microsoft Word)

    Unsupported file formats (e.g. .exe, .bin, .png) are safely ignored.

    Args:
        file_path: Path to the document file.

    Returns:
        List of Document objects (may be multiple for multi-page PDFs).
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        logger.warning(f"Document path does not exist or is not a regular file: {file_path}")
        return []

    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.info(f"Skipping unsupported file format '{ext}': {path.name}")
        return []

    try:
        if ext in (".txt", ".md"):
            raw_text = load_text_file(path).strip()
            if not raw_text:
                return []
            return [
                Document(
                    text=raw_text,
                    metadata={
                        "source": str(path.resolve()),
                        "filename": path.name,
                        "file_type": ext,
                        "page": 1,
                        "char_count": len(raw_text),
                    },
                )
            ]

        elif ext == ".pdf":
            return load_pdf_file(path)

        elif ext == ".docx":
            return load_docx_file(path)

    except Exception as exc:
        logger.error(f"Unexpected error while loading document {path}: {exc}")
        return []

    return []


def load_directory(dir_path: Union[str, Path], recursive: bool = True) -> List[Document]:
    """Scan a directory and load all supported documents safely.

    Args:
        dir_path: Path to the documents folder.
        recursive: Whether to search subdirectories recursively.

    Returns:
        List of Document objects found and extracted.
    """
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        logger.warning(f"Documents directory does not exist: {root}")
        return []

    documents: List[Document] = []
    pattern = "**/*" if recursive else "*"

    for entry in sorted(root.glob(pattern)):
        if entry.is_file() and not entry.name.startswith("."):
            docs = load_document(entry)
            documents.extend(docs)

    logger.info(f"Loaded {len(documents)} document section(s) from {root}")
    return documents


def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Split text into manageable, overlapping chunks preserving word boundaries.

    Educational Background on Chunking:
    ----------------------------------
    - Fixed Character Windows: Simply slicing `text[i : i + chunk_size]` can cut
      words in half (e.g. "super" + "vised").
    - Word Boundary Awareness: We slide by `(chunk_size - chunk_overlap)`, but
      find the nearest clean whitespace boundary to avoid word truncation.
    - Overlap: Overlapping ensures context is not severed across splits.

    Args:
        text: Source document text to split.
        metadata: Base metadata dictionary to attach to every chunk.
        chunk_size: Maximum character length per chunk (must be > 0).
        chunk_overlap: Number of overlapping characters between chunks (0 <= overlap < chunk_size).

    Returns:
        List of chunk dictionaries with 'text', 'chunk_id', 'chunk_index', and 'metadata'.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be non-negative and strictly less than chunk_size ({chunk_size}), got {chunk_overlap}"
        )

    base_meta = dict(metadata) if metadata else {}
    doc_identifier = base_meta.get("filename", "doc")
    page_identifier = base_meta.get("page")

    # If the entire text fits in one chunk, return immediately
    if len(clean_text) <= chunk_size:
        chunk_id = f"{doc_identifier}_p{page_identifier}_c1" if page_identifier else f"{doc_identifier}_c1"
        return [
            {
                "chunk_id": chunk_id,
                "chunk_index": 1,
                "total_chunks": 1,
                "text": clean_text,
                "char_count": len(clean_text),
                "metadata": {
                    **base_meta,
                    "chunk_id": chunk_id,
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    step = chunk_size - chunk_overlap
    raw_chunks: List[str] = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If not at the very end of text, attempt to snap 'end' back to a whitespace
        if end < text_len:
            boundary = clean_text.rfind(" ", start, end)
            newline_boundary = clean_text.rfind("\n", start, end)
            best_boundary = max(boundary, newline_boundary)
            # Only snap if the boundary doesn't collapse the chunk excessively
            if best_boundary > start + (chunk_size // 2):
                end = best_boundary

        chunk_slice = clean_text[start:end].strip()
        if chunk_slice:
            raw_chunks.append(chunk_slice)

        if end >= text_len:
            break

        prev_start = start
        start = start + step

        # Snap start back to the beginning of the word if it sliced into the middle of a word
        if 0 < start < text_len and not clean_text[start - 1].isspace():
            prev_boundary = max(clean_text.rfind(" ", 0, start), clean_text.rfind("\n", 0, start))
            if prev_boundary != -1 and (start - (prev_boundary + 1)) <= chunk_overlap:
                start = prev_boundary + 1

        # If start hasn't progressed past previous slice or ended, advance safely
        if start <= prev_start or start >= end:
            start = end

    total_chunks = len(raw_chunks)
    processed_chunks: List[Dict[str, Any]] = []

    for idx, c_text in enumerate(raw_chunks, 1):
        if page_identifier is not None:
            chunk_id = f"{doc_identifier}_p{page_identifier}_c{idx}"
        else:
            chunk_id = f"{doc_identifier}_c{idx}"

        chunk_dict = {
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "text": c_text,
            "char_count": len(c_text),
            "metadata": {
                **base_meta,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
            },
        }
        processed_chunks.append(chunk_dict)

    return processed_chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Split a collection of documents into chunks.

    Args:
        documents: List of Document instances.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlapping character count.

    Returns:
        Flattened list of chunk dictionaries across all documents.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        chunks = chunk_text(
            text=doc.text,
            metadata=doc.metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


# ==============================================================================
# Indexing CLI Entry Point
# ==============================================================================

def index_documents(
    documents_dir: Union[str, Path] = "documents",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """Scan documents directory, build embeddings, and update the local index.

    Can be invoked directly via CLI:
    `python -m app.rag.loader`

    Args:
        documents_dir: Path to directory containing user documents.
        chunk_size: Target size per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Dictionary summarizing the indexing results.
    """
    # Ensure UTF-8 output on Windows consoles
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Import retriever lazily to avoid circular imports
    from app.rag.retriever import LocalRetriever

    doc_path = Path(documents_dir)
    print(f"\n========================================================")
    print(f"        AURA RAG Pipeline - Document Indexer            ")
    print(f"========================================================")
    try:
        print(f"Scanning directory: {doc_path}")
    except UnicodeEncodeError:
        print("Scanning documents directory...")

    if not doc_path.exists():
        print(f"Creating empty documents directory at: {doc_path}")
        doc_path.mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    documents = load_directory(doc_path)
    print(f"Found {len(documents)} document section(s) to index.")

    if not documents:
        print("No supported documents found (.txt, .md, .pdf, .docx).")
        print("Place documents in the 'documents/' folder and run this again.")
        return {"documents_loaded": 0, "chunks_indexed": 0, "total_in_store": 0}

    # 2. Chunk text
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Generated {len(chunks)} text chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")

    # 3. Build & update vector index
    retriever = LocalRetriever()
    retriever.add_documents(chunks)
    retriever.save_index()

    print(f"Successfully updated local vector store at: {retriever.vector_store.storage_path}")
    print(f"Total active chunks in index: {len(retriever.vector_store)}")
    print(f"========================================================\n")

    return {
        "documents_loaded": len(documents),
        "chunks_indexed": len(chunks),
        "total_in_store": len(retriever.vector_store),
    }


def index_file(
    file_path: Union[str, Path],
    retriever: Optional[Any] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """Load a single document file, chunk it, embed, and update the vector store index.

    Reuses the existing document loader, chunker, and LocalRetriever vector store.

    Args:
        file_path: Path to the target document.
        retriever: Optional LocalRetriever instance (uses default LocalRetriever if None).
        chunk_size: Character length per chunk.
        chunk_overlap: Overlapping characters count.

    Returns:
        Dictionary summarizing the indexing result:
        {"documents_loaded": int, "chunks_indexed": int, "total_in_store": int}
    """
    from app.rag.retriever import LocalRetriever

    path = Path(file_path)
    docs = load_document(path)
    if not docs:
        logger.warning(f"No document content could be extracted from {path}")
        return {"documents_loaded": 0, "chunks_indexed": 0, "total_in_store": 0}

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.warning(f"No text chunks generated for document {path}")
        return {"documents_loaded": len(docs), "chunks_indexed": 0, "total_in_store": 0}

    r = retriever or LocalRetriever()
    indexed_count = r.add_documents(chunks)
    r.save_index()

    return {
        "documents_loaded": len(docs),
        "chunks_indexed": indexed_count,
        "total_in_store": len(r.vector_store),
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "documents"
    index_documents(target_dir)
