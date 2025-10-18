# src/vectordb/search.py
"""
Module này cung cấp các chức năng tìm kiếm nâng cao trên Qdrant,
bao gồm tìm kiếm vector đơn giản và các chiến lược phức tạp hơn như MMR
để tăng sự đa dạng của kết quả.
"""
from typing import List, Any
from qdrant_client.models import ScoredPoint
from .store import VectorStore

def search(query: str, vector_store: VectorStore, top_k: int = 5, threshold: float = 0.3) -> List[ScoredPoint]:
    """
    Thực hiện tìm kiếm vector trong collection.

    Args:
        query (str): Câu truy vấn tìm kiếm.
        vector_store (VectorStore): Kho vector để tìm kiếm.
        top_k (int): Số lượng kết quả hàng đầu cần trả về.
        threshold (float): Ngưỡng điểm tương đồng tối thiểu.

    Returns:
        List[ScoredPoint]: Danh sách các kết quả tìm thấy.
    """
    print(f"🔍 Đang tìm kiếm với truy vấn: '{query[:50]}...'")
    
    # 1. Embed câu truy vấn
    query_vector = vector_store.embedding_model.encode(query)
    
    # 2. Thực hiện tìm kiếm trong Qdrant
    search_results = vector_store.client.search(
        collection_name=vector_store.collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True  # Lấy cả payload (nội dung, nguồn,...)
    )
    
    print(f"  - Tìm thấy {len(search_results)} kết quả phù hợp.")
    return search_results
