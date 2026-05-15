"""
Data Loader Module — handles document ingestion, statistics, and chunking.
Documents are uploaded via the GUI (no hard-coded datasets).
Uses a simple recursive character text splitter to avoid heavy dependencies.
"""

import os
from typing import List, Dict, Any

from langchain_core.documents import Document


def load_documents_from_uploaded_files(uploaded_files, save_dir: str) -> List[Document]:
    """
    Reads uploaded file objects (from Streamlit) and returns LangChain Document objects.
    Also persists copies to `save_dir` so they can be inspected later.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        save_dir: Directory to save copies of the uploaded files.

    Returns:
        List of LangChain Document objects.
    """
    os.makedirs(save_dir, exist_ok=True)
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        # Read content
        content = uploaded_file.read().decode("utf-8", errors="replace")

        # Save a copy to disk
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Create LangChain Document
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": uploaded_file.name,
                    "size_bytes": len(content.encode("utf-8")),
                },
            )
        )

    return documents


def get_dataset_stats(documents: List[Document]) -> Dict[str, Any]:
    """
    Compute summary statistics for the loaded documents.

    Returns:
        Dictionary with keys: num_documents, total_characters, total_size_bytes,
        avg_document_length, min_document_length, max_document_length.
    """
    if not documents:
        return {
            "num_documents": 0,
            "total_characters": 0,
            "total_size_bytes": 0,
            "avg_document_length": 0,
            "min_document_length": 0,
            "max_document_length": 0,
        }

    lengths = [len(doc.page_content) for doc in documents]
    sizes = [doc.metadata.get("size_bytes", len(doc.page_content.encode("utf-8"))) for doc in documents]

    return {
        "num_documents": len(documents),
        "total_characters": sum(lengths),
        "total_size_bytes": sum(sizes),
        "avg_document_length": sum(lengths) // len(lengths),
        "min_document_length": min(lengths),
        "max_document_length": max(lengths),
    }


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively split text into chunks using paragraph, sentence, and word boundaries.
    This avoids importing langchain_text_splitters which pulls torch and may conflict.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _recursive_split(text_piece: str, sep_index: int) -> List[str]:
        if len(text_piece) <= chunk_size:
            return [text_piece] if text_piece.strip() else []

        if sep_index >= len(separators):
            # Hard split as a last resort
            parts = []
            for i in range(0, len(text_piece), chunk_size - chunk_overlap):
                part = text_piece[i : i + chunk_size]
                if part.strip():
                    parts.append(part)
            return parts

        sep = separators[sep_index]
        if not sep:
            return _recursive_split(text_piece, sep_index + 1)

        segments = text_piece.split(sep)
        result = []
        current = ""

        for segment in segments:
            candidate = (current + sep + segment) if current else segment
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    result.append(current)
                if len(segment) > chunk_size:
                    result.extend(_recursive_split(segment, sep_index + 1))
                    current = ""
                else:
                    current = segment

        if current.strip():
            result.append(current)

        return result

    raw_chunks = _recursive_split(text, 0)

    # Apply overlap by prepending tail of previous chunk
    final_chunks: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            chunk = prev_tail + chunk
        final_chunks.append(chunk)

    return final_chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    chunked: List[Document] = []
    for doc in documents:
        text_chunks = _split_text(doc.page_content, chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(text_chunks):
            chunked.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                    },
                )
            )
    return chunked
