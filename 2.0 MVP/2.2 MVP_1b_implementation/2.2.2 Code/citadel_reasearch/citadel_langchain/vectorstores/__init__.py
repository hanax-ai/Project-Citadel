
"""
Vector stores for Project Citadel LangChain integration.

This module provides vector stores for storing and retrieving embeddings.
"""

from .base import BaseVectorStore
from .inmemory import InMemoryVectorStore
from .faiss import FAISSVectorStore

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "FAISSVectorStore",
]
