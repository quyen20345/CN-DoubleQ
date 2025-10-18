# src/chunking/semantic_similarity.py
import re
import numpy as np
from typing import List
from src.embedding.model import get_embedding_model

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

def chunk(text: str, percentile_threshold: int = 90) -> List[str]:
    """
    Splits text based on semantic changes between adjacent sentences.
    """
    print("...Using strategy: Semantic Similarity Chunking")
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return sentences

    embedder = get_embedding_model()
    embeddings = np.array(embedder.encode(sentences))

    similarities = [_cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]
    
    split_threshold = np.percentile(similarities, percentile_threshold)

    chunks = []
    current_chunk_start_idx = 0
    for i, sim in enumerate(similarities):
        if sim < split_threshold:
            chunk_content = " ".join(sentences[current_chunk_start_idx : i+1])
            chunks.append(chunk_content)
            current_chunk_start_idx = i + 1

    chunks.append(" ".join(sentences[current_chunk_start_idx:]))
    return [c for c in chunks if c.strip()]
