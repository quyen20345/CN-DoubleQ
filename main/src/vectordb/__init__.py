"""
VectorDB package for interacting with Qdrant.

This package provides a high-level interface for indexing and searching documents.
"""

# Expose the main functionalities for easy importing
from .store import get_vector_store
from .indexing import index_documents
from .search import search_in_collection
from .qdrant import VectorStore
from .client import get_qdrant_client

__all__ = [
    "get_vector_store",
    "index_documents",
    "search_in_collection",
    "VectorStore",
    "get_qdrant_client",
]
