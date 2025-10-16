import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def adaptive_chunking(text: str, doc_type: str = "technical") -> List[str]:
    """
    Chunking thích ứng dựa trên loại tài liệu.
    """
    from main.src.chunking.semantic_chunking import semantic_chunking
    from main.src.chunking.sliding_window_chunking import sliding_window_chunking
    from main.src.chunking.chunking import chunking
    
    if doc_type == "technical":
        # Tài liệu kỹ thuật: chunks lớn hơn, giữ structure
        return chunking(text, chunk_size=600, chunk_overlap=150)
    elif doc_type == "qa":
        # Q&A: chunks nhỏ hơn, overlap nhiều
        return sliding_window_chunking(text, window_size=400, stride=200)
    elif doc_type == "narrative":
        # Văn bản tự sự: semantic chunking
        return semantic_chunking(text, target_size=500)
    else:
        # Default
        return chunking(text, chunk_size=512, chunk_overlap=126)