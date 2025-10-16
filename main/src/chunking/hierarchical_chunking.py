# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def hierarchical_chunking(text: str) -> List[Tuple[str, str, int]]:
    """
    Hierarchical chunking: tạo chunks ở nhiều levels.
    Returns: List of (chunk, parent_id, level)
    
    Level 0: Section level (theo heading)
    Level 1: Paragraph level
    Level 2: Sentence level (nếu cần)
    """
    chunks_with_hierarchy = []
    
    # Level 0: Tách theo heading
    sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
    
    section_id = 0
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            heading = sections[i]
            content = sections[i + 1]
            
            # Section chunk
            section_chunk = heading + "\n" + content
            chunks_with_hierarchy.append((section_chunk.strip(), None, 0))
            
            # Level 1: Paragraphs trong section
            paragraphs = re.split(r'\n\n+', content)
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:
                    chunks_with_hierarchy.append((para, section_id, 1))
            
            section_id += 1
    
    return chunks_with_hierarchy