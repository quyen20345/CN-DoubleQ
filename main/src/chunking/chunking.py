# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 100) -> List[str]:
    """
    Intelligent chunking với semantic preservation.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    chunks = []
    
    # Bước 1: Tách theo structure (headings, tables, lists)
    sections = _split_by_structure(text)
    
    # Bước 2: Chunk từng section
    for section_type, section_content in sections:
        if section_type == 'table':
            # Bảng được giữ nguyên hoặc split theo rows
            table_chunks = _chunk_table(section_content, chunk_size)
            chunks.extend(table_chunks)
        elif section_type == 'list':
            # List được chunk thông minh
            list_chunks = _chunk_list(section_content, chunk_size, chunk_overlap)
            chunks.extend(list_chunks)
        else:
            # Text thường
            text_chunks = _chunk_text(section_content, chunk_size, chunk_overlap)
            chunks.extend(text_chunks)
    
    # Bước 3: Post-processing
    cleaned_chunks = _post_process_chunks(chunks)
    
    return cleaned_chunks


def _split_by_structure(text: str) -> List[Tuple[str, str]]:
    """
    Tách text theo cấu trúc: heading, table, list, text.
    Returns: List of (type, content) tuples
    """
    sections = []
    
    # Pattern để nhận diện các structures
    patterns = {
        'heading': r'^#{1,6}\s+.+$',
        'table': r'\|.+\|[\s\S]*?\n(?:\|[\s\S]*?\n)+',
        'list': r'(?:^\s*[-*•]\s+.+$\n?)+',
    }
    
    current_pos = 0
    structure_matches = []
    
    # Tìm tất cả structures
    for struct_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.MULTILINE):
            structure_matches.append((match.start(), match.end(), struct_type, match.group()))
    
    # Sort theo position
    structure_matches.sort(key=lambda x: x[0])
    
    # Extract sections
    for start, end, struct_type, content in structure_matches:
        # Add text trước structure (nếu có)
        if current_pos < start:
            text_before = text[current_pos:start].strip()
            if text_before:
                sections.append(('text', text_before))
        
        # Add structure
        sections.append((struct_type, content))
        current_pos = end
    
    # Add text cuối cùng
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            sections.append(('text', remaining))
    
    # Nếu không có structure nào, coi toàn bộ là text
    if not sections:
        sections.append(('text', text))
    
    return sections


def _chunk_table(table_text: str, max_size: int) -> List[str]:
    """
    Chunk bảng thông minh: giữ header, split rows nếu cần.
    """
    lines = table_text.strip().split('\n')
    
    if len(lines) <= 2:  # Chỉ có header
        return [table_text]
    
    # Header (2 dòng đầu: tiêu đề + separator)
    header = '\n'.join(lines[:2])
    rows = lines[2:]
    
    # Nếu bảng nhỏ, giữ nguyên
    if len(table_text) <= max_size * 1.5:
        return [table_text]
    
    # Split thành nhiều sub-tables
    chunks = []
    current_chunk = header + "\n"
    
    for row in rows:
        if len(current_chunk) + len(row) + 1 > max_size:
            chunks.append(current_chunk.strip())
            current_chunk = header + "\n" + row + "\n"
        else:
            current_chunk += row + "\n"
    
    if current_chunk.strip() != header:
        chunks.append(current_chunk.strip())
    
    return chunks


def _chunk_list(list_text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk list items, giữ nguyên cấu trúc list.
    """
    # Tách thành các items
    items = re.findall(r'^\s*[-*•]\s+.+$', list_text, re.MULTILINE)
    
    if not items:
        return [list_text]
    
    chunks = []
    current_chunk = ""
    
    for item in items:
        if len(current_chunk) + len(item) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = item + "\n"
        else:
            current_chunk += item + "\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk text thường với semantic boundaries.
    """
    # Sử dụng RecursiveCharacterTextSplitter với separators tối ưu
    separators = [
        "\n\n\n",  # Multiple newlines
        "\n\n",    # Paragraph break
        "\n",      # Line break
        ". ",      # Sentence end
        "! ",      # Exclamation
        "? ",      # Question
        "; ",      # Semicolon
        ", ",      # Comma
        " ",       # Space
        ""         # Character
    ]
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        keep_separator=True,
    )
    
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _post_process_chunks(chunks: List[str]) -> List[str]:
    """
    Post-processing: lọc noise, merge chunks quá nhỏ.
    """
    processed = []
    
    for chunk in chunks:
        # Skip chunks quá ngắn hoặc là noise
        if len(chunk) < 50:
            continue
        
        if _is_noise_chunk(chunk):
            continue
        
        # Clean whitespace
        chunk = re.sub(r'\n{3,}', '\n\n', chunk)
        chunk = chunk.strip()
        
        processed.append(chunk)
    
    # Merge chunks quá nhỏ với chunk trước đó
    merged = []
    for i, chunk in enumerate(processed):
        if i == 0:
            merged.append(chunk)
        elif len(chunk) < 100 and len(merged[-1]) < 400:
            # Merge với chunk trước
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    
    return merged


def _is_noise_chunk(chunk: str) -> bool:
    """Kiểm tra chunk có phải là noise không."""
    # Chỉ có số, dấu cách, ký tự đặc biệt
    if re.match(r'^[\d\s\-_|\.]+$', chunk):
        return True
    
    # Quá ít từ có nghĩa
    words = [w for w in chunk.split() if len(w) > 2]
    if len(words) < 5:
        return True
    
    # Repeated pattern (watermark, header)
    lines = chunk.split('\n')
    if len(lines) > 2 and len(set(lines)) < len(lines) * 0.5:
        return True
    
    return False

