# src/rag_system/retriever.py
"""
Module này chứa Trình truy xuất lai (Hybrid Retriever), kết hợp
sức mạnh của tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa (Vector Search)
để lấy ra các tài liệu liên quan nhất.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict

from src.vectordb.store import VectorStore

class HybridRetriever:
    """
    Kết hợp BM25 và Vector Search để tìm kiếm thông tin.
    """
    def __init__(self, vector_store: VectorStore, corpus_data: List[Dict[str, str]]):
        """
        Khởi tạo retriever.
        
        Args:
            vector_store: Instance của VectorStore (Qdrant).
            corpus_data: Danh sách các dict, mỗi dict chứa {'content': str, 'source': str}.
        """
        self.vector_store = vector_store
        self.corpus_data = corpus_data
        
        # Chỉ lấy phần 'content' để tạo corpus cho BM25
        tokenized_corpus = [doc["content"].split(" ") for doc in self.corpus_data]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Khởi tạo BM25 index thành công với {len(self.corpus_data)} tài liệu.")

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
        """
        Thực hiện Reciprocal Rank Fusion để kết hợp các danh sách kết quả.
        """
        rrf_scores = {}
        for doc_list in ranked_lists:
            for rank, doc_id in enumerate(doc_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        return rrf_scores

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        """
        Thực hiện tìm kiếm lai và trả về top_k kết quả tốt nhất.
        
        Args:
            query: Câu hỏi hoặc chuỗi truy vấn.
            top_k: Số lượng tài liệu cần trả về.
            
        Returns:
            Danh sách các tài liệu (dạng dict) liên quan nhất.
        """
        print(f"\n🔍 Bắt đầu tìm kiếm lai cho query: '{query[:100]}...'")
        
        # 1. Tìm kiếm bằng Vector Search (Semantic)
        vector_results = self.vector_store.search(query, top_k=top_k, threshold=0.2)
        vector_doc_ids = [res.payload['content'] for res in vector_results if res.payload]
        print(f"  - Vector Search tìm thấy {len(vector_doc_ids)} kết quả.")

        # 2. Tìm kiếm bằng BM25 (Keyword)
        tokenized_query = query.split(" ")
        bm25_results = self.bm25.get_top_n(tokenized_query, [doc["content"] for doc in self.corpus_data], n=top_k)
        bm25_doc_ids = [content for content in bm25_results]
        print(f"  - BM25 Search tìm thấy {len(bm25_doc_ids)} kết quả.")
        
        # 3. Kết hợp kết quả bằng RRF
        fused_scores = self._reciprocal_rank_fusion([vector_doc_ids, bm25_doc_ids])
        
        # Sắp xếp các tài liệu dựa trên điểm RRF
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # Lấy top_k tài liệu duy nhất từ corpus gốc
        final_results = []
        seen_content = set()
        for doc_id in sorted_doc_ids:
            if doc_id not in seen_content:
                # Tìm tài liệu gốc trong corpus
                original_doc = next((doc for doc in self.corpus_data if doc["content"] == doc_id), None)
                if original_doc:
                    final_results.append(original_doc)
                    seen_content.add(doc_id)
            if len(final_results) >= top_k:
                break
        
        print(f"  - Sau khi kết hợp, trả về {len(final_results)} tài liệu tốt nhất.")
        return final_results
