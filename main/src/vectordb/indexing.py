from typing import List, Dict, Any
from .store import get_vector_store

def index_documents(
    collection_name: str,
    embedding_model: Any,
    documents: List[Dict[str, Any]]
):
    """
    High-level function to handle the indexing of documents into a vector store.

    Args:
        collection_name (str): The target Qdrant collection name.
        embedding_model (Any): The model for creating embeddings.
        documents (List[Dict[str, Any]]): A list of documents to index.
                                          Each document is a dictionary.
    """
    if not documents:
        print("No documents provided to index.")
        return

    print(f"Starting indexing for {len(documents)} documents into '{collection_name}'...")
    
    vector_store = get_vector_store(collection_name, embedding_model)
    vector_store.upsert_data(payloads=documents)
    
    print("✅ Indexing process completed.")
