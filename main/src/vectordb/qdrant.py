import os
import re
from dotenv import load_dotenv
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    HnswConfigDiff, Filter, FieldCondition, MatchValue
)

load_dotenv()

QDRANT_CLIENT = QdrantClient(
    host=os.getenv("QDRANT_HOST"), 
    port=os.getenv("QDRANT_PORT"), 
    timeout=int(os.getenv("QDRANT_TIMEOUT"))
)

class VectorStore:
    def __init__(self, collection_name, dense_model):
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        """Tạo collection với cấu hình tối ưu cho search."""
        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=self.dense_embedding_model.get_dimension(),
                    distance=Distance.COSINE,
                )
            },
            # Tối ưu HNSW config
            hnsw_config=HnswConfigDiff(
                m=16,  # Số kết nối tối đa
                ef_construct=100,  # Chất lượng xây dựng index
            ),
        ) 

    def recreate_collection(self):
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)
        self._create_collection()

    def enable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(m=16),
        ) 

    def disable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(m=0),
        ) 

    def get_collection_info(self):
        return QDRANT_CLIENT.get_collection(self.collection_name)

    def _embed_contents(self, chunk_contents):
        """Embed với batch processing."""
        if isinstance(chunk_contents, str):
            return [self.dense_embedding_model.encode(chunk_contents)]
        return self.dense_embedding_model.encode(chunk_contents)
        
    def _upsert_data(self, points):
        """Upsert với batch size tối ưu."""
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"✓ Indexed batch {i // BATCH_SIZE + 1}/{(len(points) - 1) // BATCH_SIZE + 1}")

    def insert_data(self, payload_keys, payload_values, embedding_indices=[0]):
        """
        Insert data với metadata phong phú hơn.
        
        Args:
            payload_keys: Danh sách các key cho payload
            payload_values: Danh sách các giá trị tương ứng
            embedding_indices: Indices của các trường dùng để tạo embedding
        """
        if not payload_values:
            return
        
        if isinstance(payload_values[0], str):
            contents = payload_values
        elif isinstance(payload_values[0], list):
            contents = [
                "\n".join(str(v[i]) for i in embedding_indices)
                for v in payload_values
            ]
        else:
            raise ValueError("Unsupported payload_values format")

        dense_embeddings = self._embed_contents(contents)

        self.upsert_data(
            dense_embeddings=dense_embeddings,
            payload_keys=payload_keys,
            payload_values=payload_values
        )

    def upsert_data(self, dense_embeddings, payload_keys, payload_values):
        """Upsert với metadata tăng cường."""
        points = []
        
        for i in range(len(dense_embeddings)):
            # Tạo payload
            if isinstance(payload_values[i], list):
                payload = dict(zip(payload_keys, payload_values[i]))
            else:
                payload = dict(zip(payload_keys, [payload_values[i]]))
            
            # Thêm metadata bổ sung (với error handling)
            try:
                content = payload.get('content', '')
                payload['content_length'] = len(content)
                payload['has_table'] = '|' in content and content.count('|') > 5
                payload['has_list'] = bool(re.search(r'^\s*[-*]\s+', content, re.MULTILINE))
            except Exception as e:
                # Nếu có lỗi khi thêm metadata, vẫn tiếp tục
                pass
            
            # Tạo point
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense_vector": dense_embeddings[i]},
                    payload=payload
                )
            )

        self._upsert_data(points)

    def _search_dense(self, query, top_k):
        """
        Search dense vectors.
        
        Args:
            query: Query text
            top_k: Số kết quả trả về
        """
        query_vector = self.dense_embedding_model.encode(query)
        
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,  
            using="dense_vector",  
            with_payload=True,
            limit=top_k,
        )
        
        return results.points 
    
    def search_dense(self, query, top_k=5, threshold=0.3):
        """
        Dense vector search với filtering.
        
        Args:
            query: Query text
            top_k: Số kết quả tối đa
            threshold: Ngưỡng score tối thiểu
        """
        results = self._search_dense(query, top_k * 2)  # Lấy nhiều hơn để filter
        
        # Filter theo threshold
        filtered = [point for point in results if point.score >= threshold]
        
        # Trả về top_k kết quả tốt nhất
        return filtered[:top_k]
    
    def search_with_source_filter(self, query, source_name, top_k=5, threshold=0.3):
        """Search với filter theo source cụ thể."""
        query_vector = self.dense_embedding_model.encode(query)
        
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense_vector",
            with_payload=True,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_name)
                    )
                ]
            )
        )
        
        return [point for point in results.points if point.score >= threshold]
    
    def search(self, query, top_k=5, threshold=0.3):
        """
        Main search function với các cải tiến.
        """
        results = self.search_dense(query, top_k, threshold)
        return results
    
    def hybrid_search(self, query, top_k=5, threshold=0.25):
        """
        Hybrid search: kết hợp vector search và keyword matching.
        Hữu ích cho các câu hỏi có thuật ngữ kỹ thuật cụ thể.
        """
        # Vector search
        vector_results = self.search_dense(query, top_k * 2, threshold=threshold)
        
        # Extract keywords từ query
        keywords = re.findall(r'\b\w{4,}\b', query.lower())
        
        # Re-score kết quả dựa trên keyword matching
        scored_results = []
        for point in vector_results:
            content = point.payload.get('content', '').lower()
            
            # Đếm số keywords xuất hiện
            keyword_matches = sum(1 for kw in keywords if kw in content)
            keyword_score = keyword_matches / max(len(keywords), 1)
            
            # Kết hợp điểm số
            combined_score = point.score * 0.7 + keyword_score * 0.3
            
            scored_results.append((combined_score, point))
        
        # Sắp xếp và trả về top_k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [point for _, point in scored_results[:top_k]]