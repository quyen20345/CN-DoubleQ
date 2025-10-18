# src/data_processing/pdf_parser.py
"""
Module này chứa class `PDFMarkdownConverter` chịu trách nhiệm cho tất cả logic
trích xuất nội dung từ file PDF và chuyển đổi nó thành định dạng Markdown.
Bao gồm xử lý văn bản, bảng, hình ảnh, và các yếu tố cấu trúc khác.
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Tuple

class PDFMarkdownConverter:
    """
    Class chuyên dụng để chuyển đổi PDF sang Markdown, giữ lại cấu trúc và định dạng.
    """
    def __init__(self):
        self.heading_pattern = r'^(\d+(?:\.\d+)*)\s+(.+)$'
        self.figure_caption_patterns = [
            r'^Hình\s+\d+[\.:]\s*(.+)$',
            r'^Figure\s+\d+[\.:]\s*(.+)$',
        ]
        self.global_image_counter = 0

    def _get_file_title(self, pdf_path: str) -> str:
        """Tạo tiêu đề chính cho file Markdown từ tên file PDF."""
        filename = Path(pdf_path).stem
        return f"# {filename.replace('_', ' ').title()}"

    def _detect_text_style(self, span: Dict) -> Tuple[bool, bool]:
        """Phát hiện chữ đậm và chữ nghiêng từ thông tin font."""
        font_name = span.get('font', '').lower()
        flags = span.get('flags', 0)
        is_bold = 'bold' in font_name or (flags & 2**4)
        is_italic = 'italic' in font_name or (flags & 2**1)
        return is_bold, is_italic

    def _format_text(self, text: str, is_bold: bool, is_italic: bool) -> str:
        """Áp dụng định dạng Markdown cho văn bản."""
        text = text.strip()
        if not text:
            return ""
        if is_bold and is_italic:
            return f"***{text}***"
        if is_bold:
            return f"**{text}**"
        if is_italic:
            return f"*{text}*"
        return text

    def _extract_tables(self, page) -> List[Dict]:
        """Sử dụng pdfplumber để trích xuất bảng một cách hiệu quả."""
        try:
            import pdfplumber
            with pdfplumber.open(page.parent.name) as pdf:
                plumber_page = pdf.pages[page.number]
                tables = plumber_page.find_tables()
                extracted = []
                for tbl in tables:
                    content_raw = tbl.extract()
                    if not content_raw:
                        continue
                    # Chuyển đổi list của list thành bảng Markdown
                    header = "| " + " | ".join(map(str, content_raw[0])) + " |"
                    separator = "| " + " | ".join(["---"] * len(content_raw[0])) + " |"
                    body = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in content_raw[1:]])
                    md_table = f"{header}\n{separator}\n{body}"
                    extracted.append({'content': md_table, 'bbox': tbl.bbox, 'y_pos': tbl.bbox[1]})
                return extracted
        except Exception as e:
            print(f"Lỗi khi trích xuất bảng ở trang {page.number}: {e}")
            return []


    def _extract_images(self, page, images_dir: Path) -> List[Dict]:
        """Trích xuất và lưu hình ảnh từ một trang."""
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                img_rect = page.get_image_bbox(img)
                
                # Bỏ qua ảnh nhỏ (có thể là icon hoặc noise)
                if img_rect.width < 50 or img_rect.height < 50:
                    continue
                
                self.global_image_counter += 1
                image_filename = f"image_{page.number}_{self.global_image_counter}.{base_image['ext']}"
                image_path = images_dir / image_filename
                
                with open(image_path, "wb") as f_img:
                    f_img.write(image_bytes)
                
                images.append({
                    'filename': image_filename,
                    'bbox': img_rect,
                    'y_pos': img_rect.y0
                })
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_index} ở trang {page.number}: {e}")
        return images


    def _process_page_elements(self, page, images_dir: Path) -> str:
        """Xử lý và sắp xếp tất cả các thành phần trên trang."""
        elements = []
        # Lấy văn bản
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0: # text block
                for l in b['lines']:
                    elements.append({'type': 'text', 'content': l, 'bbox': fitz.Rect(l['bbox']), 'y_pos': l['bbox'][1]})
        
        # Lấy ảnh
        for img in self._extract_images(page, images_dir):
            elements.append({'type': 'image', 'content': img, 'bbox': img['bbox'], 'y_pos': img['y_pos']})
            
        # Lấy bảng
        for tbl in self._extract_tables(page):
             elements.append({'type': 'table', 'content': tbl['content'], 'bbox': fitz.Rect(tbl['bbox']), 'y_pos': tbl['y_pos']})

        # Sắp xếp tất cả các element theo vị trí dọc (y_pos)
        elements.sort(key=lambda el: el['y_pos'])
        
        # Tạo nội dung Markdown
        md_content = []
        for el in elements:
            if el['type'] == 'text':
                line_text = ""
                for s in el['content']['spans']:
                    is_bold, is_italic = self._detect_text_style(s)
                    line_text += self._format_text(s['text'], is_bold, is_italic)
                md_content.append(line_text)
            elif el['type'] == 'image':
                md_content.append(f"\n![{el['content']['filename']}](images/{el['content']['filename']})\n")
            elif el['type'] == 'table':
                 md_content.append(f"\n{el['content']}\n")

        return "\n".join(md_content)

    def convert(self, pdf_path: str, output_dir: str) -> Tuple[str, int]:
        """
        Hàm chính thực hiện việc chuyển đổi.
        Trả về tuple (md_content, image_count)
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_file = output_dir.parent / "main.md"
        images_dir = output_dir

        images_dir.mkdir(parents=True, exist_ok=True)
        self.global_image_counter = 0

        doc = fitz.open(pdf_path)
        all_md_content = [self._get_file_title(str(pdf_path))]
        
        print(f"Bắt đầu xử lý {pdf_path.name}...")
        for i, page in enumerate(doc):
            print(f" - Trang {i+1}/{len(doc)}")
            page_content = self._process_page_elements(page, images_dir)
            all_md_content.append(page_content)

        doc.close()

        final_md = "\n".join(all_md_content)
        # Hậu xử lý để dọn dẹp file Markdown
        final_md = re.sub(r'\n{3,}', '\n\n', final_md).strip()
        
        output_file.write_text(final_md, encoding='utf-8')
        print(f"✅ Chuyển đổi thành công: {output_file}")
        
        # Trả về nội dung markdown và số lượng ảnh
        return final_md, self.global_image_counter
