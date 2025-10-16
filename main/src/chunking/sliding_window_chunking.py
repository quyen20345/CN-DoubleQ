# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def sliding_window_chunking(text: str, window_size: int = 512, 
                            stride: int = 256) -> List[str]:
    """
    Sliding window chunking: overlap lớn để đảm bảo không mất context.
    """
    from main.src.chunking.chunking import _is_noise_chunk
    if len(text) <= window_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + window_size
        
        # Tìm boundary tốt (end of sentence)
        if end < len(text):
            # Tìm dấu câu gần nhất
            for boundary in ['. ', '! ', '? ', '\n\n', '\n', ' ']:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos != -1:
                    end = boundary_pos + len(boundary)
                    break
        
        chunk = text[start:end].strip()
        if chunk and not _is_noise_chunk(chunk):
            chunks.append(chunk)
        
        start += stride
    
    return chunks