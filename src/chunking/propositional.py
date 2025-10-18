# src/chunking/propositional.py
from typing import List
from src.llm.client import get_llm
from . import recursive_character # Use as fallback

def chunk(text: str, chunk_size: int = 256) -> List[str]:
    """
    Uses an LLM to extract propositions (atomic pieces of information),
    then groups them into chunks.
    """
    print("...Using strategy: Propositional (Agentic) Chunking")
    llm = get_llm(temperature=0.0)
    
    prompt = f"""Decompose the following text into a list of simple propositions, one per line.
Each proposition must be a self-contained, factual statement.

ORIGINAL TEXT:
---
{text[:2000]}
---

PROPOSITIONS:
"""
    try:
        response = llm.invoke(prompt)
        propositions = [p.strip() for p in response.split('\n') if p.strip()]
    except Exception as e:
        print(f"Error extracting propositions: {e}. Falling back to recursive chunking.")
        return recursive_character.chunk(text)

    if not propositions:
        return recursive_character.chunk(text)

    chunks = []
    current_chunk = ""
    for prop in propositions:
        if len(current_chunk.split()) + len(prop.split()) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = prop
        else:
            current_chunk += f" {prop}"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks
