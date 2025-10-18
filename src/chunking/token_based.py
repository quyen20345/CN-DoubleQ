# src/chunking/token_based.py
from typing import List
from langchain.text_splitter import TokenTextSplitter

def chunk(text: str, chunk_size: int = 256, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text based on the number of tokens.
    Useful for ensuring chunks do not exceed an LLM's token limit.
    """
    print("...Using strategy: Token-Based Splitting")
    if not isinstance(text, str) or not text.strip():
        return []
    
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
