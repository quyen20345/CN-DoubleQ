from typing import Any, Dict
from .qdrant import VectorStore

# A cache for vector store instances to avoid re-initialization
_vector_store_cache: Dict[str, VectorStore] = {}

def get_vector_store(collection_name: str, embedding_model: Any) -> VectorStore:
    """
    Factory function to get a VectorStore instance.
    
    Caches instances based on collection name to ensure a single instance
    per collection is used throughout the application.

    Args:
        collection_name (str): The name of the collection.
        embedding_model (Any): The embedding model instance.

    Returns:
        VectorStore: An instance of the VectorStore.
    """
    if collection_name not in _vector_store_cache:
        print(f"Initializing VectorStore for collection: '{collection_name}'")
        _vector_store_cache[collection_name] = VectorStore(
            collection_name=collection_name,
            dense_model=embedding_model
        )
    return _vector_store_cache[collection_name]
