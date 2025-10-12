# -*- coding: utf-8 -*-
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 100) -> list[str]:
    """
    Phân nhỏ văn bản với chiến lược intelligent chunking:
    - Ưu tiên giữ nguyên các bảng
    - Không cắt ngang các heading
    - Overlap để đảm bảo context liên tục
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    chunks = []
    
    # Bước 1: Tách các phần lớn dựa trên heading
    sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
    
    current_section = ""
    current_heading = ""
    
    for i, section in enumerate(sections):
        # Nếu là heading
        if re.match(r'^#{1,6}\s+', section):
            # Lưu section trước đó nếu có
            if current_section.strip():
                chunks.extend(_chunk_section(current_heading + "\n" + current_section, 
                                             chunk_size, chunk_overlap))
            current_heading = section
            current_section = ""
        else:
            current_section += section
    
    # Xử lý section cuối cùng
    if current_section.strip():
        chunks.extend(_chunk_section(current_heading + "\n" + current_section, 
                                     chunk_size, chunk_overlap))
    
    # Bước 2: Lọc và làm sạch chunks
    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Giữ chunks có độ dài hợp lý và có nội dung
        if len(chunk) >= 50 and not _is_noise_chunk(chunk):
            cleaned_chunks.append(chunk)
    
    return cleaned_chunks


def _chunk_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk một section cụ thể, bảo toàn bảng và cấu trúc."""
    # Kiểm tra xem có bảng không
    if '|' in text and text.count('|') > 10:
        # Nếu có bảng, tách bảng ra riêng
        return _chunk_with_tables(text, chunk_size, chunk_overlap)
    else:
        # Chunk bình thường
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)


def _chunk_with_tables(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Xử lý text có chứa bảng Markdown."""
    chunks = []
    
    # Pattern để nhận diện bảng markdown
    table_pattern = r'(\|.+\|[\s\S]*?\n(?:\|[\s\S]*?\n)+)'
    
    parts = re.split(table_pattern, text)
    
    current_chunk = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Nếu là bảng
        if part.startswith('|') and part.count('|') > 10:
            # Lưu chunk hiện tại nếu có
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Thêm bảng như một chunk riêng (kèm heading nếu có)
            table_chunk = part
            
            # Nếu bảng quá lớn, chia nhỏ theo rows
            if len(table_chunk) > chunk_size * 2:
                table_rows = table_chunk.split('\n')
                header = '\n'.join(table_rows[:2])  # Header + separator
                
                sub_table = header + "\n"
                for row in table_rows[2:]:
                    if len(sub_table) + len(row) > chunk_size:
                        chunks.append(sub_table.strip())
                        sub_table = header + "\n" + row + "\n"
                    else:
                        sub_table += row + "\n"
                
                if sub_table.strip():
                    chunks.append(sub_table.strip())
            else:
                chunks.append(table_chunk)
        else:
            # Text thường, tích lũy vào chunk hiện tại
            if len(current_chunk) + len(part) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += "\n" + part
    
    # Lưu chunk cuối
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _is_noise_chunk(chunk: str) -> bool:
    """Kiểm tra xem chunk có phải là noise không."""
    # Loại bỏ chunks chỉ chứa:
    # - Số trang
    # - Header/footer lặp lại
    # - Chỉ có ký tự đặc biệt
    
    if re.match(r'^[\d\s\-_|]+$', chunk):
        return True
    
    # Chunks quá ngắn chỉ có 1-2 từ
    words = chunk.split()
    if len(words) < 5:
        return True
    
    # Chunks chỉ là watermark hoặc repeated header
    if chunk.count('\n') < 2 and len(set(chunk.split())) < 3:
        return True
    
    return False


def adaptive_chunking(text: str, doc_type: str = "technical") -> list[str]:
    """
    Chunking thích ứng dựa trên loại tài liệu.
    
    Args:
        text: Văn bản cần chunk
        doc_type: Loại tài liệu ("technical", "narrative", "mixed")
    
    Returns:
        List các chunks
    """
    if doc_type == "technical":
        # Tài liệu kỹ thuật: chunks lớn hơn, overlap nhiều hơn
        return chunking(text, chunk_size=600, chunk_overlap=150)
    elif doc_type == "narrative":
        # Văn bản tự sự: chunks nhỏ hơn, overlap ít hơn
        return chunking(text, chunk_size=400, chunk_overlap=80)
    else:
        # Mặc định
        return chunking(text, chunk_size=512, chunk_overlap=100)