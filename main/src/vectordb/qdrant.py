import os
from dotenv import load_dotenv
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, HnswConfigDiff

load_dotenv()

QDRANT_CLIENT = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"), 
                             timeout=int(os.getenv("QDRANT_TIMEOUT")))

class VectorStore:
    def __init__(self, collection_name, dense_model):
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=self.dense_embedding_model.get_dimension(),
                    distance=Distance.COSINE,
                )
            },

            hnsw_config=HnswConfigDiff(
                m=16,
            ),
        ) 

    def recreate_collection(self):
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)

        self._create_collection()

    def enable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=16,
            ),
        ) 

    def disable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=0,
            ),
        ) 

    def get_collection_info(self):
        return QDRANT_CLIENT.get_collection(self.collection_name)

    def _embed_contents(self, chunk_contents):
        dense_embeddings = self.dense_embedding_model.encode(chunk_contents)
        return dense_embeddings
        
    def _upsert_data(self, points):
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Upserted batch {i // BATCH_SIZE + 1} of {len(points) // BATCH_SIZE + 1}")

    def insert_data(self, payload_keys, payload_values, embedding_indices=[0]):
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
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense_vector": dense_embeddings[i],
                },
                payload=dict(zip(payload_keys, payload_values[i] if isinstance(payload_values[i], list) else [payload_values[i]]))
            )
            for i in range(len(dense_embeddings))
        ]

        self._upsert_data(points)

    def _search_dense(self, query, top_k):
        query_vector = self.dense_embedding_model.encode(query)
    
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector,  
            using="dense_vector",  
            with_payload=True,
            limit=top_k,
        )
        
        return results.points 
    
    def search_dense(self, query, top_k=3, threshold=0.3):
        results = self._search_dense(query, top_k)
        return [point for point in results if point.score >= threshold]
    
    def search(self, query, top_k=5, threshold=0.3):
        results = self.search_dense(query, top_k, threshold)
        return results