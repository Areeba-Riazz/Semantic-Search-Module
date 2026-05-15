"""
Vector Store Manager — creates, persists, and queries FAISS or Chroma
vector stores through LangChain.  The user selects the store type in the GUI.
"""

import os
import shutil
import time
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma


def create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    store_type: str = "FAISS",
    persist_directory: str = "./Vector_Store/store",
) -> object:
    """
    Create a vector store from chunked documents.

    Args:
        documents: List of LangChain Document objects (chunked).
        embeddings: HuggingFaceEmbeddings instance.
        store_type: 'FAISS' or 'Chroma'.
        persist_directory: Directory to persist the vector store.

    Returns:
        The vector store object (FAISS or Chroma).
    """
    # Clean existing store directory to avoid stale data
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    start_time = time.time()

    if store_type == "FAISS":
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(persist_directory)
    elif store_type == "Chroma":
        vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_directory,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

    elapsed = time.time() - start_time
    return vector_store, elapsed


def query_vector_store(
    vector_store,
    query: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Run a semantic similarity search against the vector store.

    Args:
        vector_store: A FAISS or Chroma vector store object.
        query: The user's natural-language query.
        top_k: Number of results to return.

    Returns:
        List of (Document, similarity_score) tuples, highest relevance first.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results


def load_vector_store(
    store_type: str,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str,
) -> object:
    """
    Load a previously persisted vector store from disk.
    """
    if store_type == "FAISS":
        return FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    elif store_type == "Chroma":
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
