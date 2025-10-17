# main/src/chunking/chunking.py
# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunking(text: str, chunk_size: int = 600, chunk_overlap: int = 150) -> List[str]:
    """
    Hàm chunking thông minh, ưu tiên giữ lại cấu trúc và ngữ nghĩa của tài liệu kỹ thuật.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Bước 1: Tách văn bản thành các khối cấu trúc (tiêu đề, bảng, danh sách, văn bản thường)
    # Pattern để tìm các khối markdown.
    # Nâng cấp pattern để bắt cả khối mã và xử lý tiêu đề tốt hơn.
    pattern = re.compile(
        r'(^#{1,6}\s.*$)|'          # 1: Headings
        r'(\|.*\|(?:\n\|.*\|)+)|'   # 2: Tables
        r'((?:^\s*[-*•]\s.*(?:\n|$))+)|'  # 3: Lists
        r'(```[\s\S]*?```)',        # 4: Code blocks
        re.MULTILINE
    )

    chunks = []
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        
        # Lấy phần văn bản thường nằm giữa hai khối cấu trúc
        plain_text_part = text[last_end:start].strip()
        if plain_text_part:
            text_chunks = _chunk_plain_text(plain_text_part, chunk_size, chunk_overlap)
            chunks.extend(text_chunks)
            
        # Lấy khối cấu trúc (bảng, danh sách, mã)
        structured_block = match.group().strip()
        
        # Nếu khối cấu trúc quá lớn, vẫn chia nhỏ nó một cách thông minh
        if len(structured_block) > chunk_size * 1.5:
            if structured_block.startswith('|'): # Là bảng
                 chunks.extend(_chunk_table(structured_block, chunk_size))
            elif structured_block.startswith(('`', '*', '-', '•')): # Là code hoặc list
                 chunks.extend(_chunk_plain_text(structured_block, chunk_size, chunk_overlap))
            else: # Là tiêu đề dài hoặc trường hợp khác
                 chunks.append(structured_block)
        else:
            chunks.append(structured_block)
            
        last_end = end
    
    # Xử lý phần văn bản cuối cùng
    remaining_text = text[last_end:].strip()
    if remaining_text:
        text_chunks = _chunk_plain_text(remaining_text, chunk_size, chunk_overlap)
        chunks.extend(text_chunks)

    # Bước 3: Xử lý hậu kỳ
    return _post_process_chunks(chunks)


def _chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Sử dụng RecursiveCharacterTextSplitter để chia nhỏ văn bản thường."""
    if not text:
        return []
    
    # Các dấu hiệu ngắt câu tốt nhất cho văn bản kỹ thuật
    separators = [
        "\n\n\n", "\n\n", "\n", 
        ". ", "! ", "? ", "; ", ", ", 
        " ", ""
    ]
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )
    
    return splitter.split_text(text)

def _chunk_table(table_md: str, max_size: int) -> List[str]:
    """Chia nhỏ bảng nhưng luôn giữ lại dòng header."""
    lines = table_md.strip().split('\n')
    if len(lines) <= 2:
        return [table_md] # Bảng không có nội dung
        
    header = f"{lines[0]}\n{lines[1]}"
    rows = lines[2:]
    
    if len(table_md) <= max_size:
        return [table_md]
        
    # Chia thành các bảng con
    sub_tables = []
    current_rows = []
    current_len = len(header)
    
    for row in rows:
        if current_len + len(row) > max_size and current_rows:
            sub_tables.append(f"{header}\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header)
        
        current_rows.append(row)
        current_len += len(row)
        
    if current_rows:
        sub_tables.append(f"{header}\n" + "\n".join(current_rows))
        
    return sub_tables

def _is_noise(text: str, min_words: int = 5, min_chars: int = 20) -> bool:
    """Kiểm tra một chunk có phải là nhiễu không (ví dụ: số trang, header)."""
    text = text.strip()
    # Loại bỏ các chunk quá ngắn
    if len(text) < min_chars:
        return True
    # Loại bỏ các chunk có quá ít từ
    if len(re.findall(r'\w+', text)) < min_words:
        return True
    # Loại bỏ các chunk chỉ chứa ký tự lặp lại hoặc ký tự đặc biệt
    if re.fullmatch(r'[\s\d\W_]+', text):
        return True
    return False

def _post_process_chunks(chunks: List[str], min_chunk_size: int = 50) -> List[str]:
    """Lọc bỏ nhiễu và hợp nhất các chunk quá nhỏ."""
    # Lọc bỏ nhiễu
    filtered_chunks = [c.strip() for c in chunks if not _is_noise(c)]
    
    if not filtered_chunks:
        return []

    # Hợp nhất các chunk nhỏ
    merged_chunks = []
    current_chunk = filtered_chunks[0]
    
    for i in range(1, len(filtered_chunks)):
        next_chunk = filtered_chunks[i]
        # Nếu chunk hiện tại quá nhỏ, hợp nhất với chunk tiếp theo
        if len(current_chunk) < min_chunk_size:
            current_chunk += "\n\n" + next_chunk
        else:
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
            
    merged_chunks.append(current_chunk) # Thêm chunk cuối cùng
    
    return merged_chunks