"""
Embedding Manager — wraps Hugging Face embedding models via LangChain.
No model is hard-coded; the user selects from the GUI.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import EMBEDDING_MODELS


def get_available_models() -> List[str]:
    """Return the display names of all available embedding models."""
    return list(EMBEDDING_MODELS.keys())


def get_embedding_model(model_display_name: str) -> HuggingFaceEmbeddings:
    """
    Instantiate and return a LangChain HuggingFaceEmbeddings object
    for the selected model.

    Args:
        model_display_name: Display name (key from EMBEDDING_MODELS dict).

    Returns:
        HuggingFaceEmbeddings instance ready for use.

    Raises:
        ValueError: If the model name is not in the available list.
    """
    if model_display_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model '{model_display_name}'. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )

    model_id = EMBEDDING_MODELS[model_display_name]
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
