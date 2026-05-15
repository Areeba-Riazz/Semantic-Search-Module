"""
Configuration module for the AI Research Assistant — Phase 1: Semantic Search.
Defines paths, available models, vector DB options, and default parameters.
"""

import os

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "Vector_Store")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# ──────────────────────────────────────────────
# Available Hugging Face Embedding Models
# ──────────────────────────────────────────────
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
}

# ──────────────────────────────────────────────
# Available Vector Databases
# ──────────────────────────────────────────────
VECTOR_DB_OPTIONS = ["FAISS", "Chroma"]

# ──────────────────────────────────────────────
# Default Parameters
# ──────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_VECTOR_DB = "FAISS"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# ──────────────────────────────────────────────
# Supported File Extensions
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf"]
