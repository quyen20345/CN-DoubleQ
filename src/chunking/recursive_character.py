# src/chunking/recursive_character.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text recursively based on a list of characters.
    This is a basic, fast, and effective method.
    """
    print("...Using strategy: Recursive Character Splitting")
    if not isinstance(text, str) or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    return splitter.split_text(text)
