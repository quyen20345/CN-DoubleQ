from typing import List, Any
from .store import get_vector_store

def search_in_collection(
    collection_name: str,
    embedding_model: Any,
    query: str,
    top_k: int = 5,
    search_method: str = 'hybrid',
    threshold: float = 0.3
) -> List[Any]:
    """
    High-level function to perform a search in a vector store collection.

    Args:
        collection_name (str): The collection to search in.
        embedding_model (Any): The embedding model instance.
        query (str): The search query.
        top_k (int): The number of results to return.
        search_method (str): The search strategy to use ('hybrid', 'mmr').
        threshold (float): The minimum score threshold for results.

    Returns:
        List[Any]: A list of search results (ScoredPoint objects).
    """
    print(f"Performing '{search_method}' search...")
    
    vector_store = get_vector_store(collection_name, embedding_model)
    
    results = vector_store.search(
        query=query,
        top_k=top_k,
        threshold=threshold,
        method=search_method
    )
    
    print(f"Found {len(results)} results.")
    return results
