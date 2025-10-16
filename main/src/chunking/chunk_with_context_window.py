from typing import List

def chunk_with_context_window(text: str, chunk_size: int = 512, 
                               context_size: int = 100) -> List[str]:
    """
    Chunking với context window: mỗi chunk có thêm context từ chunks liền kề.
    """
    from main.src.chunking.chunking import chunking
    # Chunk bình thường trước
    base_chunks = chunking(text, chunk_size, chunk_overlap=50)
    
    # Thêm context
    chunks_with_context = []
    for i, chunk in enumerate(base_chunks):
        # Context trước
        prev_context = ""
        if i > 0:
            prev_context = base_chunks[i-1][-context_size:] + "\n[...]\n"
        
        # Context sau
        next_context = ""
        if i < len(base_chunks) - 1:
            next_context = "\n[...]\n" + base_chunks[i+1][:context_size]
        
        # Kết hợp
        chunk_with_ctx = prev_context + chunk + next_context
        chunks_with_context.append(chunk_with_ctx.strip())
    
    return chunks_with_context