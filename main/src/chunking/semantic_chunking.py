# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunking(text: str, target_size: int = 512) -> List[str]:
    """
    Semantic chunking: chunk dựa trên ý nghĩa, không cắt ngang câu/đoạn.
    """
    # Tách thành paragraphs
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Nếu paragraph quá dài, split thành sentences
        if len(para) > target_size * 1.5:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sent in sentences:
                if len(current_chunk) + len(sent) > target_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk += " " + sent if current_chunk else sent
        else:
            # Paragraph bình thường
            if len(current_chunk) + len(para) > target_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks