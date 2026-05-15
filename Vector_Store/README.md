# Vector Store Module

This module manages **vector database creation, persistence, and retrieval**.

## Key File

- **`vector_store_manager.py`** — Creates and queries FAISS or Chroma vector stores through LangChain.

## Supported Backends

| Backend | Description |
|---|---|
| **FAISS** | Fast in-memory similarity search (Facebook AI). Persisted to local files. |
| **Chroma** | Persistent, metadata-rich vector database. Stored in a local directory. |

## Sub-directories (auto-created at runtime)

- `faiss_store/` — FAISS index files
- `chroma_store/` — Chroma database files

The vector store type is selected through the GUI — neither is hard-coded.
