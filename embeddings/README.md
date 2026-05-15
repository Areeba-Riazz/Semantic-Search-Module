# Embeddings Module

This module provides the **embedding generation** layer for the Semantic Search system.

## Key File

- **`embedding_manager.py`** — Wraps Hugging Face sentence-transformer models via LangChain's `HuggingFaceEmbeddings`.

## Supported Models

| Display Name | Hugging Face Model ID |
|---|---|
| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 |
| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 |
| multi-qa-MiniLM-L6-cos-v1 | sentence-transformers/multi-qa-MiniLM-L6-cos-v1 |
| paraphrase-MiniLM-L6-v2 | sentence-transformers/paraphrase-MiniLM-L6-v2 |
| all-distilroberta-v1 | sentence-transformers/all-distilroberta-v1 |

Models are selected dynamically through the GUI — none are hard-coded.
