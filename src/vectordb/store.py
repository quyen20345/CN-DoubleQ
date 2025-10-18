# src/vectordb/store.py
"""
Module này định nghĩa class `VectorStore` để quản lý các collection trong Qdrant.
Nó đóng gói logic tạo collection, xóa, và các thao tác quản trị khác.
"""

from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
from .client import get_qdrant_client
from ..embedding.model import EmbeddingModel

class VectorStore:
    """
    Lớp quản lý một collection cụ thể trong Qdrant.
    """
    def __init__(self, collection_name: str, embedding_model: EmbeddingModel):
        self.client = get_qdrant_client()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Tự động tạo collection nếu chưa tồn tại
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """
        Tạo collection với cấu hình tối ưu nếu nó chưa tồn tại.
        """
        try:
            # collection_exists nhanh hơn get_collection
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_model.get_dimension(),
                        distance=Distance.COSINE
                    ),
                    # Cấu hình HNSW để cân bằng giữa tốc độ và độ chính xác
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
                )
                print(f"✅ Collection '{self.collection_name}' đã được tạo.")
        except Exception as e:
            # Xử lý trường hợp collection đã tồn tại do race condition
            if "already exists" not in str(e):
                 print(f"Lỗi khi tạo collection '{self.collection_name}': {e}")
                 raise

    def recreate_collection(self):
        """Xóa và tạo lại collection. Hữu ích khi muốn làm mới dữ liệu."""
        print(f"⚠️ Đang xóa và tạo lại collection '{self.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_model.get_dimension(),
                distance=Distance.COSINE
            ),
        )
        print(f"✅ Collection '{self.collection_name}' đã được làm mới.")

    def get_collection_info(self) -> dict:
        """Lấy thông tin về collection, ví dụ: số lượng vector."""
        try:
            return self.client.get_collection(self.collection_name).model_dump()
        except Exception as e:
            print(f"Không thể lấy thông tin collection '{self.collection_name}': {e}")
            return {}
