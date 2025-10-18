# src/vectordb/indexer.py
"""
Module này chứa logic để chunking, embedding, và tải dữ liệu vào Qdrant.
"""

import uuid
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Generator, Tuple
from qdrant_client.models import PointStruct

from .store import VectorStore
from src.chunking import get_chunking_strategy

load_dotenv()

def _batch_generator(data: List, batch_size: int) -> Generator[List, None, None]:
    """Tạo ra các khối (batches) dữ liệu từ một danh sách."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def index_documents(extracted_data: Dict[str, str], vector_store: VectorStore) -> List[Dict[str, str]]:
    """
    Xử lý và index dữ liệu, đồng thời trả về corpus cho BM25.
    
    Returns:
        List[Dict[str, str]]: Corpus chứa tất cả các chunk để sử dụng cho BM25.
    """
    print("🔄 Bắt đầu quá trình chunking và indexing...")
    
    chunking_strategy_name = os.getenv("CHUNKING_STRATEGY", "recursive_char")
    chunk_text = get_chunking_strategy(chunking_strategy_name)
    
    vector_store.recreate_collection()
    
    all_points = []
    corpus_for_bm25 = []
    total_chunks = 0
    
    for doc_name, content in extracted_data.items():
        print(f"  - Đang xử lý tài liệu: {doc_name}")
        
        raw_chunks = chunk_text(content)
        if not raw_chunks:
            print(f"    - ⚠️ Không tạo được chunk nào cho {doc_name}.")
            continue
            
        chunks = [f"[{doc_name}] {chunk}" for chunk in raw_chunks]
        
        embeddings = vector_store.embedding_model.encode(chunks)
        
        for chunk, emb in zip(chunks, embeddings):
            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={"content": chunk, "source": doc_name}
                )
            )
            corpus_for_bm25.append({"content": chunk, "source": doc_name})
        
        print(f"    - Đã tạo {len(chunks)} chunks.")
        total_chunks += len(chunks)

    if all_points:
        BATCH_SIZE = 128
        print(f"\n🚀 Đang tải {len(all_points)} điểm dữ liệu lên Qdrant theo từng khối {BATCH_SIZE} điểm...")
        
        for batch in _batch_generator(all_points, BATCH_SIZE):
            vector_store.client.upsert(
                collection_name=vector_store.collection_name,
                points=batch,
                wait=True
            )
            print(f"   - Đã tải lên thành công {len(batch)} điểm.")
    
    print(f"✅ Hoàn thành indexing! Tổng cộng {total_chunks} chunks.")
    return corpus_for_bm25

