# -*- coding: utf-8 -*-
import sys
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Lỗi: langchain hoặc tiktoken chưa được cài. Vui lòng chạy: pip install langchain tiktoken")
    sys.exit(1)

def chunking(text: str) -> list[str]:
    """Phân nhỏ văn bản thành các đoạn có kích thước phù hợp."""
    if not isinstance(text, str) or not text.strip():
        return []
        
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    # Lọc bỏ các chunk quá ngắn để tránh làm nhiễu kết quả tìm kiếm
    return [chunk for chunk in chunks if len(chunk.strip()) >= 50]