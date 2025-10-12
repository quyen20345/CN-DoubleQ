import os
import re
from dotenv import load_dotenv
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    HnswConfigDiff, Filter, FieldCondition, MatchValue, SearchRequest
)

load_dotenv()

QDRANT_CLIENT = QdrantClient(
    host=os.getenv("QDRANT_HOST"), 
    port=os.getenv("QDRANT_PORT"), 
    timeout=int(os.getenv("QDRANT_TIMEOUT"))
)

class VectorStore:
    """Vector store được tối ưu cho RAG accuracy."""
    
    def __init__(self, collection_name, dense_model):
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        """Tạo collection với cấu hình tối ưu."""
        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=self.dense_embedding_model.get_dimension(),
                    distance=Distance.COSINE,
                )
            },
            hnsw_config=HnswConfigDiff(
                m=32,  # Tăng từ 16 -> 32 cho accuracy cao hơn
                ef_construct=200,  # Tăng từ 100 -> 200
                full_scan_threshold=10000,
            ),
            optimizers_config={
                "indexing_threshold": 10000,
            }
        )
        print(f"✅ Tạo collection '{self.collection_name}' với cấu hình cao")

    def recreate_collection(self):
        """Xóa và tạo lại collection."""
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)
        self._create_collection()

    def get_collection_info(self):
        """Lấy thông tin collection."""
        return QDRANT_CLIENT.get_collection(self.collection_name)

    def _embed_contents(self, chunk_contents):
        """Embed với batch processing."""
        if isinstance(chunk_contents, str):
            return [self.dense_embedding_model.encode(chunk_contents)]
        return self.dense_embedding_model.encode(chunk_contents)
        
    def _upsert_data(self, points):
        """Upsert với batch size tối ưu."""
        BATCH_SIZE = 100
        total_batches = (len(points) - 1) // BATCH_SIZE + 1
        
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            batch_num = i // BATCH_SIZE + 1
            print(f"  ✓ Indexed batch {batch_num}/{total_batches}")

    def insert_data(self, payload_keys, payload_values, embedding_indices=[0]):
        """Insert data với metadata phong phú."""
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
            
            # Thêm metadata để filter và rank tốt hơn
            try:
                content = payload.get('content', '')
                
                # Metadata cơ bản
                payload['content_length'] = len(content)
                payload['word_count'] = len(content.split())
                
                # Phát hiện loại content
                payload['has_table'] = '|' in content and content.count('|') > 5
                payload['has_list'] = bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE))
                payload['has_heading'] = bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE))
                payload['has_code'] = bool(re.search(r'```|`\w+`', content))
                
                # Phát hiện keywords kỹ thuật (IoT, Smart Home)
                tech_keywords = [
                    'mqtt', 'iot', 'sensor', 'actuator', 'gateway', 
                    'protocol', 'device', 'smart', 'automation',
                    'zigbee', 'bluetooth', 'wifi', 'api', 'server'
                ]
                content_lower = content.lower()
                payload['tech_score'] = sum(1 for kw in tech_keywords if kw in content_lower)
                
                # Tính information density (số từ kỹ thuật / tổng số từ)
                words = content_lower.split()
                if words:
                    tech_word_count = sum(1 for w in words if any(kw in w for kw in tech_keywords))
                    payload['info_density'] = tech_word_count / len(words)
                else:
                    payload['info_density'] = 0.0
                
            except Exception as e:
                # Fallback nếu có lỗi
                payload['content_length'] = 0
                payload['word_count'] = 0
                payload['tech_score'] = 0
                payload['info_density'] = 0.0
            
            # Tạo point
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense_vector": dense_embeddings[i]},
                    payload=payload
                )
            )

        self._upsert_data(points)

    def _search_dense(self, query, top_k, score_threshold=0.0):
        """Dense vector search cơ bản."""
        query_vector = self.dense_embedding_model.encode(query)
        
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,  
            using="dense_vector",  
            with_payload=True,
            limit=top_k,
            score_threshold=score_threshold,
        )
        
        return results.points
    
    def search_dense(self, query, top_k=5, threshold=0.3):
        """Dense search với filtering."""
        results = self._search_dense(query, top_k * 2, score_threshold=threshold * 0.8)
        
        # Post-filter và re-rank
        filtered = []
        for point in results:
            # Filter theo threshold chính xác
            if point.score >= threshold:
                filtered.append(point)
        
        return filtered[:top_k]
    
    def search_with_metadata_boost(self, query, top_k=5, threshold=0.25):
        """Search với boosting dựa trên metadata."""
        # Lấy nhiều kết quả hơn để re-rank
        results = self._search_dense(query, top_k * 3, score_threshold=threshold * 0.7)
        
        # Re-rank với metadata boosting
        scored_results = []
        for point in results:
            base_score = point.score
            
            # Boosting factors
            tech_boost = point.payload.get('tech_score', 0) * 0.02  # +2% per tech keyword
            density_boost = point.payload.get('info_density', 0) * 0.1  # +10% max
            
            # Penalty cho content quá ngắn hoặc quá dài
            word_count = point.payload.get('word_count', 0)
            if word_count < 30:
                length_factor = 0.9  # -10%
            elif word_count > 500:
                length_factor = 0.95  # -5%
            else:
                length_factor = 1.0
            
            # Boost cho content có table/list (thường chứa thông tin có cấu trúc)
            structure_boost = 0
            if point.payload.get('has_table', False):
                structure_boost += 0.05
            if point.payload.get('has_list', False):
                structure_boost += 0.03
            
            # Tính final score
            final_score = (base_score + tech_boost + density_boost + structure_boost) * length_factor
            
            if final_score >= threshold:
                scored_results.append((final_score, point))
        
        # Sort và return top_k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [point for _, point in scored_results[:top_k]]
    
    def hybrid_search(self, query, top_k=5, threshold=0.25):
        """Hybrid search: vector + keyword matching."""
        # Vector search
        vector_results = self._search_dense(query, top_k * 2, score_threshold=threshold * 0.7)
        
        # Extract keywords từ query
        keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        # Re-score với keyword matching
        scored_results = []
        for point in vector_results:
            content = point.payload.get('content', '').lower()
            
            # Keyword matching score
            content_words = set(re.findall(r'\b\w{3,}\b', content))
            overlap = len(keywords & content_words)
            keyword_score = overlap / max(len(keywords), 1)
            
            # Keyword proximity score (keywords xuất hiện gần nhau)
            proximity_score = self._calculate_keyword_proximity(content, keywords)
            
            # Combined score
            combined_score = (
                point.score * 0.6 +  # Vector similarity
                keyword_score * 0.25 +  # Keyword overlap
                proximity_score * 0.15  # Keyword proximity
            )
            
            if combined_score >= threshold:
                scored_results.append((combined_score, point))
        
        # Sort và return
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [point for _, point in scored_results[:top_k]]
    
    def _calculate_keyword_proximity(self, text: str, keywords: set) -> float:
        """Tính độ gần nhau của keywords trong text."""
        if not keywords:
            return 0.0
        
        words = text.split()
        positions = []
        
        for i, word in enumerate(words):
            if word.lower() in keywords:
                positions.append(i)
        
        if len(positions) < 2:
            return 0.3 if positions else 0.0
        
        # Tính khoảng cách trung bình
        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_distance = sum(distances) / len(distances)
        
        # Score cao hơn khi keywords gần nhau
        # Distance 1-5: score 1.0, distance 10: score 0.5, distance 50+: score ~0
        proximity_score = 1.0 / (1.0 + avg_distance / 5)
        return min(proximity_score, 1.0)
    
    def mmr_search(self, query, top_k=5, threshold=0.25, lambda_param=0.7):
        """
        Maximal Marginal Relevance search để đa dạng hóa kết quả.
        Tránh trả về các chunks quá giống nhau.
        """
        # Lấy nhiều candidates
        candidates = self._search_dense(query, top_k * 4, score_threshold=threshold * 0.7)
        
        if not candidates:
            return []
        
        # Embed query
        query_vector = self.dense_embedding_model.encode(query)
        
        selected = []
        remaining = list(candidates)
        
        # Chọn document đầu tiên (score cao nhất)
        selected.append(remaining.pop(0))
        
        # Chọn các documents tiếp theo dựa trên MMR
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score (với query)
                relevance = candidate.score
                
                # Diversity score (so với các docs đã chọn)
                max_similarity = 0
                candidate_vector = candidate.vector.get('dense_vector', [])
                
                for selected_point in selected:
                    selected_vector = selected_point.vector.get('dense_vector', [])
                    similarity = self._cosine_similarity(candidate_vector, selected_vector)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((mmr, candidate))
            
            # Chọn candidate có MMR cao nhất
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                best_candidate = mmr_scores[0][1]
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _cosine_similarity(self, vec1, vec2):
        """Tính cosine similarity giữa 2 vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def contextual_search(self, query, context_queries: List[str], 
                         top_k=5, threshold=0.25):
        """
        Search với context từ nhiều queries liên quan.
        Useful cho multi-hop questions.
        """
        # Search cho query chính
        main_results = self.search_with_metadata_boost(query, top_k * 2, threshold)
        
        # Search cho context queries
        context_results = []
        for ctx_query in context_queries:
            results = self.search_dense(ctx_query, top_k=2, threshold=threshold)
            context_results.extend(results)
        
        # Merge và deduplicate
        all_results = {}
        for point in main_results:
            content = point.payload.get('content', '')
            all_results[content] = point
        
        for point in context_results:
            content = point.payload.get('content', '')
            if content not in all_results:
                all_results[content] = point
        
        # Re-rank dựa trên tổng relevance
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:top_k]
    
    def search(self, query, top_k=5, threshold=0.3, method='hybrid'):
        """
        Main search function với nhiều methods.
        
        Args:
            query: Search query
            top_k: Số kết quả trả về
            threshold: Score threshold
            method: 'dense', 'hybrid', 'mmr', 'metadata_boost'
        """
        if method == 'dense':
            return self.search_dense(query, top_k, threshold)
        elif method == 'hybrid':
            return self.hybrid_search(query, top_k, threshold)
        elif method == 'mmr':
            return self.mmr_search(query, top_k, threshold, lambda_param=0.7)
        elif method == 'metadata_boost':
            return self.search_with_metadata_boost(query, top_k, threshold)
        else:
            # Default: hybrid
            return self.hybrid_search(query, top_k, threshold)
    
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
    
    def multi_vector_search(self, queries: List[str], top_k=5, threshold=0.25):
        """
        Search với nhiều queries và aggregate kết quả.
        Useful cho complex questions.
        """
        all_results = {}
        
        for query in queries:
            results = self.hybrid_search(query, top_k * 2, threshold)
            
            for point in results:
                content = point.payload.get('content', '')
                
                if content in all_results:
                    # Aggregate scores (lấy max hoặc average)
                    all_results[content]['score'] = max(
                        all_results[content]['score'],
                        point.score
                    )
                    all_results[content]['count'] += 1
                else:
                    all_results[content] = {
                        'point': point,
                        'score': point.score,
                        'count': 1
                    }
        
        # Sort theo combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'] * (1 + 0.1 * x['count']),  # Boost cho docs xuất hiện nhiều lần
            reverse=True
        )
        
        return [item['point'] for item in sorted_results[:top_k]]