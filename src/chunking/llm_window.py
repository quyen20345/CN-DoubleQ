# src/chunking/llm_window.py
import re
from typing import List
from src.llm.client import get_llm

def chunk(text: str, window_size: int = 15, step_size: int = 5) -> List[str]:
    """
    Uses a sliding window and an LLM to identify semantic split points.
    """
    print("...Using strategy: LLM Window-Based Chunking")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences: return []

    split_indices = set()
    for i in range(0, len(sentences), step_size):
        window = sentences[i : i + window_size]
        if len(window) < 5: continue
        
        numbered_window_text = "\n".join(f"{idx + 1}. {sent}" for idx, sent in enumerate(window))
        prompt = f"""Analyze the following sentences and identify the most significant topic change.
Return ONLY the INTEGER number of that sentence. For example: '5'.

TEXT:
---
{numbered_window_text}
---

The number of the best sentence to split at is:"""
        try:
            llm = get_llm(temperature=0.0)
            response = llm.invoke(prompt)
            split_num = int(re.findall(r'\d+', response)[0])
            if 1 < split_num < len(window):
                split_indices.add(i + split_num - 1)
        except (ValueError, IndexError, Exception):
            continue

    if not split_indices: return [text]

    chunks, start_idx = [], 0
    for point in sorted(list(split_indices)):
        chunks.append(" ".join(sentences[start_idx:point]))
        start_idx = point
    chunks.append(" ".join(sentences[start_idx:]))
    return [c for c in chunks if c.strip()]
