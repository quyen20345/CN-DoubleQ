import re
import uuid
from typing import List, Dict, Any
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    HnswConfigDiff, Filter, FieldCondition, MatchValue
)

# Import the shared client instance
from .client import get_qdrant_client

QDRANT_CLIENT = get_qdrant_client()

class VectorStore:
    """
    Vector store implementation using Qdrant, optimized for RAG accuracy.
    This class wraps the core logic for collection management, data upsertion,
    and advanced search strategies.
    """

    def __init__(self, collection_name: str, dense_model: Any):
        """
        Initializes the VectorStore.

        Args:
            collection_name (str): The name of the collection in Qdrant.
            dense_model (Any): The model for creating embeddings (e.g., a SentenceTransformer).
                               It must have `encode` and `get_dimension` methods.
        """
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        """Creates a Qdrant collection with an optimized configuration for accuracy."""
        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=self.dense_embedding_model.get_dimension(),
                    distance=Distance.COSINE,
                )
            },
            hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
            optimizers_config={"indexing_threshold": 10000}
        )
        print(f"✅ Created collection '{self.collection_name}' with high-accuracy config.")

    def recreate_collection(self):
        """Deletes and recreates the collection."""
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)
        self._create_collection()

    def _embed_contents(self, contents: List[str]) -> List[List[float]]:
        """Embeds a list of string contents using the provided model."""
        return self.dense_embedding_model.encode(contents, batch_size=32)

    def _batch_upsert(self, points: List[PointStruct]):
        """Upserts points in batches to Qdrant."""
        QDRANT_CLIENT.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False
        )

    def _enrich_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Enriches a payload with calculated metadata for better filtering and ranking."""
        content = payload.get('content', '')
        
        # Add various metadata fields
        payload['content_length'] = len(content)
        payload['word_count'] = len(content.split())
        
        # Technical keyword scoring (example for IoT/Smart Home)
        tech_keywords = [
            'mqtt', 'iot', 'sensor', 'actuator', 'gateway', 'protocol', 
            'device', 'smart', 'automation', 'zigbee', 'bluetooth', 'wifi', 'api', 'server'
        ]
        content_lower = content.lower()
        payload['tech_score'] = sum(1 for kw in tech_keywords if kw in content_lower)
        
        return payload

    def upsert_data(self, payloads: List[Dict[str, Any]]):
        """
        Embeds and upserts data with enriched metadata.

        Args:
            payloads (List[Dict[str, Any]]): A list of dictionaries, where each
                                            must contain a 'content' key for embedding.
        """
        if not payloads:
            return

        contents = [p.get('content', '') for p in payloads]
        dense_embeddings = self._embed_contents(contents)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense_vector": emb},
                payload=self._enrich_payload(p)
            )
            for emb, p in zip(dense_embeddings, payloads)
        ]

        self._batch_upsert(points)
        print(f"  ✓ Upserted {len(points)} points to '{self.collection_name}'.")
    
    def _search_dense(self, query: str, top_k: int, score_threshold: float = 0.0) -> List[Any]:
        """Performs a basic dense vector search."""
        query_vector = self.dense_embedding_model.encode(query)
        
        return QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense_vector",
            with_payload=True,
            limit=top_k,
            score_threshold=score_threshold,
        ).points

    def hybrid_search(self, query, top_k=5, threshold=0.25):
        """Hybrid search: combines vector similarity with keyword matching."""
        vector_results = self._search_dense(query, top_k * 2, score_threshold=threshold * 0.7)
        
        keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        scored_results = []
        for point in vector_results:
            content_lower = point.payload.get('content', '').lower()
            content_words = set(re.findall(r'\b\w{3,}\b', content_lower))
            
            overlap = len(keywords & content_words)
            keyword_score = overlap / len(keywords) if keywords else 0
            
            combined_score = (point.score * 0.7) + (keyword_score * 0.3)
            
            if combined_score >= threshold:
                # Add combined score to the point object for sorting
                point.score = combined_score
                scored_results.append(point)
        
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def mmr_search(self, query, top_k=5, threshold=0.25, lambda_param=0.7):
        """Maximal Marginal Relevance (MMR) search to diversify results."""
        candidates = self._search_dense(query, top_k * 4, score_threshold=threshold * 0.7)
        if not candidates:
            return []

        selected = [candidates.pop(0)]
        while len(selected) < top_k and candidates:
            mmr_scores = []
            for candidate in candidates:
                relevance = candidate.score
                max_similarity = max(
                    self._cosine_similarity(candidate.vector['dense_vector'], s.vector['dense_vector'])
                    for s in selected
                ) if selected else 0
                
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((mmr, candidate))
            
            if not mmr_scores: break
            
            best_candidate = max(mmr_scores, key=lambda x: x[0])[1]
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        
        return selected

    def _cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        import numpy as np
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, top_k=5, threshold=0.3, method='hybrid'):
        """Main search function dispatching to different methods."""
        if method == 'mmr':
            return self.mmr_search(query, top_k, threshold)
        # Default to hybrid search
        return self.hybrid_search(query, top_k, threshold)
