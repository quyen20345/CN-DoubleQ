import fitz  # PyMuPDF
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


class PDFToMarkdownConverter:
    def __init__(self):
        # Regex patterns
        self.heading_pattern = r'^(\d+(?:\.\d+)*)\s+(.+)$'
        self.heading_with_dot = r'^(\d+)\.\s+(.+)$'
        self.header_table_bbox = (0, 0, 800, 150)
        
        # Pattern để phát hiện chú thích ảnh
        self.figure_caption_patterns = [
            r'^Hình\s+\d+[\.:]\s*(.+)$',
            r'^Figure\s+\d+[\.:]\s*(.+)$',
            r'^Fig\.\s*\d+[\.:]\s*(.+)$',
            r'^Hình\s+\d+\.\d+[\.:]\s*(.+)$',
        ]
        
        # Counter cho ảnh global
        self.global_image_counter = 0

    def get_file_title_from_pdf(self, pdf_path: str) -> str:
        """Tạo tiêu đề file từ tên file PDF"""
        filename = Path(pdf_path).stem
        if filename.startswith("Public"):
            number = filename[6:]
            return f"# Public_{number}"
        return f"# {filename}"

    def is_in_header_table_area(self, bbox: Tuple) -> bool:
        """Kiểm tra xem text block có nằm trong vùng bảng header không"""
        x0, y0, x1, y1 = bbox
        h_x0, h_y0, h_x1, h_y1 = self.header_table_bbox
        return (x0 >= h_x0 and y1 <= h_y1)

    def detect_text_style(self, span: Dict) -> Tuple[bool, bool]:
        """Phát hiện text có bold/italic không"""
        font_name = span.get('font', '').lower()
        flags = span.get('flags', 0)
        
        is_bold = (
            'bold' in font_name or 'heavy' in font_name or 
            'black' in font_name or flags & 2**4
        )
        is_italic = (
            'italic' in font_name or 'oblique' in font_name or flags & 2**1
        )
        return is_bold, is_italic

    def format_text_with_style(self, text: str, is_bold: bool, is_italic: bool) -> str:
        """Áp dụng Markdown formatting"""
        text = text.strip()
        if not text:
            return text
        if is_bold and is_italic:
            return f"***{text}***"
        elif is_bold:
            return f"**{text}**"
        elif is_italic:
            return f"*{text}*"
        return text

    def detect_heading_level(self, text: str, spans: List[Dict] = None) -> Tuple[int, str, bool]:
        """Phát hiện heading có số"""
        text = text.strip()
        # Loại bỏ markdown bold nếu có
        text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
        text = text.strip()
        
        # Pattern 1: "1.2.1.2. Quản lý thiết bị gia dụng"
        match = re.match(self.heading_pattern, text)
        if match:
            number = match.group(1)
            title = match.group(2).strip()
            
            # Phải có title, không chỉ là số
            if title:
                is_short = len(title) < 150
                no_trailing_punctuation = not title.endswith(('.', ',', ';'))
                
                if is_short or no_trailing_punctuation:
                    level = len(number.split('.'))
                    return level, title, True
        
        # Pattern 2: "1. Nội dung chính"
        match2 = re.match(self.heading_with_dot, text)
        if match2:
            title = match2.group(2).strip()
            
            # Phải có title, không chỉ là số đơn lẻ
            if title:
                is_short = len(title) < 150
                no_trailing_punctuation = not title.endswith(('.', ',', ';'))
                
                if is_short or no_trailing_punctuation:
                    return 1, title, True
        
        return 0, text, False
    
    def is_standalone_number(self, text: str) -> Tuple[bool, str]:
        """
        Kiểm tra xem text có phải chỉ là số heading đơn lẻ không
        Ví dụ: "1.", "1.1.1", "2.3", "**1.**"
        Returns: (is_number, cleaned_number)
        """
        text = text.strip()
        # Loại bỏ bold markdown
        text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text).strip()
        
        # Pattern: chỉ là số với dấu chấm (có thể có hoặc không)
        # "1", "1.", "1.1", "1.1.1", "1.1.1."
        number_pattern = r'^(\d+(?:\.\d+)*)\.?$'
        match = re.match(number_pattern, text)
        
        if match:
            return True, match.group(1)
        
        return False, text

    def detect_bold_only_heading(self, text: str, spans: List[Dict], next_line_text: str = "") -> Tuple[bool, str]:
        """Phát hiện heading chỉ có bold (không có số)"""
        text = text.strip()
        cleaned = re.sub(r'^\*\*(.+)\*\*$', r'\1', text).strip()
        
        # Điều kiện 1: Text ngắn
        if len(cleaned) > 100:
            return False, text
        
        # Điều kiện 2: Không kết thúc bằng dấu câu
        if cleaned.endswith(('.', ',', ';', '!', '?')):
            return False, text
        
        # Điều kiện 3: Toàn bộ spans phải bold
        if not spans:
            return False, text
        
        total_len = sum(len(s.get('text', '')) for s in spans)
        bold_len = 0
        
        for span in spans:
            is_bold, _ = self.detect_text_style(span)
            if is_bold:
                bold_len += len(span.get('text', ''))
        
        if total_len > 0 and (bold_len / total_len) < 0.9:
            return False, text
        
        return True, cleaned

    def is_figure_caption(self, text: str) -> bool:
        """Kiểm tra xem text có phải chú thích ảnh không"""
        text = text.strip()
        for pattern in self.figure_caption_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def is_paragraph_italic(self, line_spans: List[Dict]) -> bool:
        """Kiểm tra xem cả đoạn có phải in nghiêng không"""
        if not line_spans:
            return False
        italic_count = 0
        total_length = 0
        for span in line_spans:
            text_length = len(span.get('text', ''))
            total_length += text_length
            _, is_italic = self.detect_text_style(span)
            if is_italic:
                italic_count += text_length
        return total_length > 0 and (italic_count / total_length) > 0.8

    def extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """
        Trích xuất bảng từ page sử dụng pdfplumber
        Returns: List of {content: markdown_table, bbox: (x0,y0,x1,y1)}
        """
        table_data = []
        
        try:
            # Sử dụng pdfplumber để extract tables với bbox
            import pdfplumber
            pdf = pdfplumber.open(page.parent.name)
            plumber_page = pdf.pages[page_num]
            
            # Extract tables với bbox information
            tables = plumber_page.find_tables()
            
            for table_obj in tables:
                table_data_raw = table_obj.extract()
                
                if not table_data_raw or len(table_data_raw) < 2:
                    continue
                
                # Lấy bbox của bảng
                bbox = table_obj.bbox  # (x0, y0, x1, y1)
                
                # SKIP nếu bảng nằm trong vùng header
                if self.is_in_header_table_area(bbox):
                    continue
                
                # Chuyển sang markdown table
                md_table = self._table_to_markdown(table_data_raw)
                if md_table:
                    table_data.append({
                        'content': md_table,
                        'bbox': bbox
                    })
            
            pdf.close()
        except Exception as e:
            print(f"Warning: Could not extract tables from page {page_num}: {e}")
        
        return table_data

    def _table_to_markdown(self, table: List[List]) -> str:
        """Chuyển đổi table array sang markdown format"""
        if not table:
            return ""
        
        # Lọc bỏ rows/cells None
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cleaned_row):  # Chỉ giữ row có dữ liệu
                cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 2:
            return ""
        
        # Header row
        header = cleaned_table[0]
        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in cleaned_table[1:]:
            # Đảm bảo số cột khớp với header
            while len(row) < len(header):
                row.append("")
            md_lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(md_lines)

    def extract_page_elements(self, page, page_num: int, images_dir: Path) -> List[Dict]:
        """
        Trích xuất TẤT CẢ elements (text, image, table) theo thứ tự vị trí Y
        Returns: List of {type: 'text'|'image'|'table', content: ..., y_pos: ...}
        """
        elements = []
        
        # 1. Extract text blocks
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block['type'] != 0:
                continue
            
            bbox = block['bbox']
            if self.is_in_header_table_area(bbox):
                continue
            
            # Lấy y position (trung bình của block)
            y_pos = (bbox[1] + bbox[3]) / 2
            
            elements.append({
                'type': 'text_block',
                'block': block,
                'y_pos': y_pos,
                'bbox': bbox
            })
        
        # 2. Extract images
        images = page.get_images()
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue
                
                img_bbox = img_rects[0]
                
                if self.is_in_header_table_area(img_bbox):
                    continue
                
                base_image = page.parent.extract_image(xref)
                if base_image["width"] < 50 or base_image["height"] < 50:
                    continue
                
                y_pos = (img_bbox[1] + img_bbox[3]) / 2
                
                # Tăng global counter
                self.global_image_counter += 1
                
                # Lưu ảnh
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"image{self.global_image_counter}.{image_ext}"
                image_path = images_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                elements.append({
                    'type': 'image',
                    'filename': image_filename,
                    'y_pos': y_pos,
                    'bbox': img_bbox
                })
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")
        
        # 3. Extract tables với bbox check
        tables_data = self.extract_tables_from_page(page, page_num)
        for table_info in tables_data:
            bbox = table_info['bbox']
            y_pos = (bbox[1] + bbox[3]) / 2  # Y position của bảng
            
            elements.append({
                'type': 'table',
                'content': table_info['content'],
                'y_pos': y_pos,
                'bbox': bbox
            })
        
        # Sắp xếp theo Y position
        elements.sort(key=lambda x: x['y_pos'])
        
        return elements

    def process_text_blocks_to_lines(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Xử lý các text blocks thành lines với thông tin đầy đủ
        """
        all_lines = []
        
        for block_info in text_blocks:
            block = block_info['block']
            
            for line in block.get('lines', []):
                spans = line.get('spans', [])
                full_line = ''.join(s['text'] for s in spans).strip()
                
                if full_line:
                    all_lines.append({
                        'text': full_line,
                        'spans': spans,
                        'bbox': line.get('bbox', block_info['bbox'])
                    })
        
        return all_lines

    def extract_styled_text_from_elements(self, elements: List[Dict]) -> List[str]:
        """
        Xử lý elements theo thứ tự và trả về markdown lines
        """
        markdown_lines = []
        current_paragraph = []
        paragraph_is_italic = False
        
        # Nhóm text blocks liên tiếp
        i = 0
        while i < len(elements):
            elem = elements[i]
            
            if elem['type'] == 'image':
                # Flush paragraph trước ảnh
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if paragraph_is_italic:
                        markdown_lines.append(f"_{para_text}_")
                    else:
                        markdown_lines.append(para_text)
                    markdown_lines.append("")
                    current_paragraph = []
                    paragraph_is_italic = False
                
                # Thêm ảnh
                markdown_lines.append(f"![image](images/{elem['filename']})")
                markdown_lines.append("")
                
                i += 1
                continue
            
            elif elem['type'] == 'table':
                # Flush paragraph trước bảng
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    if paragraph_is_italic:
                        markdown_lines.append(f"_{para_text}_")
                    else:
                        markdown_lines.append(para_text)
                    markdown_lines.append("")
                    current_paragraph = []
                    paragraph_is_italic = False
                
                # Thêm bảng
                markdown_lines.append(elem['content'])
                markdown_lines.append("")
                
                i += 1
                continue
            
            elif elem['type'] == 'text_block':
                # Collect consecutive text blocks
                text_blocks = []
                while i < len(elements) and elements[i]['type'] == 'text_block':
                    text_blocks.append(elements[i])
                    i += 1
                
                # Process text blocks
                all_lines = self.process_text_blocks_to_lines(text_blocks)
                
                j = 0
                while j < len(all_lines):
                    line_info = all_lines[j]
                    full_line = line_info['text']
                    spans = line_info['spans']
                    
                    if not full_line:
                        j += 1
                        continue
                    
                    # BƯỚC 1: Kiểm tra số đơn lẻ (có thể là heading bị tách)
                    is_number, number = self.is_standalone_number(full_line)
                    
                    if is_number:
                        # Kiểm tra line tiếp theo có phải là phần title không
                        if j + 1 < len(all_lines):
                            next_line = all_lines[j + 1]['text'].strip()
                            next_spans = all_lines[j + 1]['spans']
                            
                            # Kiểm tra next_line không phải là số, không quá dài
                            is_next_number, _ = self.is_standalone_number(next_line)
                            
                            if not is_next_number and len(next_line) < 150:
                                # Gộp thành heading
                                if current_paragraph:
                                    para_text = ' '.join(current_paragraph)
                                    if paragraph_is_italic:
                                        markdown_lines.append(f"_{para_text}_")
                                    else:
                                        markdown_lines.append(para_text)
                                    markdown_lines.append("")
                                    current_paragraph = []
                                    paragraph_is_italic = False
                                
                                level = len(number.split('.'))
                                markdown_lines.append(f"{'#' * level} {next_line}")
                                markdown_lines.append("")
                                
                                # Skip cả 2 lines
                                j += 2
                                continue
                        
                        # Nếu không có line tiếp theo hoặc không match -> SKIP số đơn lẻ
                        j += 1
                        continue
                    
                    # BƯỚC 2: Kiểm tra heading có số đầy đủ
                    level, heading_text, is_numbered_heading = self.detect_heading_level(full_line, spans)
                    
                    if is_numbered_heading:
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if paragraph_is_italic:
                                markdown_lines.append(f"_{para_text}_")
                            else:
                                markdown_lines.append(para_text)
                            markdown_lines.append("")
                            current_paragraph = []
                            paragraph_is_italic = False
                        
                        markdown_lines.append(f"{'#' * level} {heading_text}")
                        markdown_lines.append("")
                        j += 1
                        continue
                    
                    # BƯỚC 3: Kiểm tra bold heading
                    next_line_text = all_lines[j+1]['text'] if j+1 < len(all_lines) else ""
                    is_bold_heading, bold_heading_text = self.detect_bold_only_heading(
                        full_line, spans, next_line_text
                    )
                    
                    if is_bold_heading:
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if paragraph_is_italic:
                                markdown_lines.append(f"_{para_text}_")
                            else:
                                markdown_lines.append(para_text)
                            markdown_lines.append("")
                            current_paragraph = []
                            paragraph_is_italic = False
                        
                        markdown_lines.append(f"### {bold_heading_text}")
                        markdown_lines.append("")
                        j += 1
                        continue
                    
                    # BƯỚC 4: Kiểm tra chú thích ảnh
                    if self.is_figure_caption(full_line):
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            if paragraph_is_italic:
                                markdown_lines.append(f"_{para_text}_")
                            else:
                                markdown_lines.append(para_text)
                            markdown_lines.append("")
                            current_paragraph = []
                            paragraph_is_italic = False
                        
                        # Thêm caption với italic
                        markdown_lines.append(f"*{full_line}*")
                        markdown_lines.append("")
                        j += 1
                        continue
                    
                    # BƯỚC 5: Xử lý paragraph bình thường
                    is_para_italic = self.is_paragraph_italic(spans)
                    
                    line_text = ""
                    for span in spans:
                        text = span['text']
                        if not text.strip():
                            line_text += text
                            continue
                        
                        if not is_para_italic:
                            is_bold, is_italic = self.detect_text_style(span)
                            formatted_text = self.format_text_with_style(text, is_bold, is_italic)
                            line_text += formatted_text
                        else:
                            line_text += text
                    
                    if line_text.strip():
                        if is_para_italic:
                            if current_paragraph and not paragraph_is_italic:
                                para_text = ' '.join(current_paragraph)
                                markdown_lines.append(para_text)
                                markdown_lines.append("")
                                current_paragraph = []
                            paragraph_is_italic = True
                            current_paragraph.append(line_text.strip())
                        else:
                            if current_paragraph and paragraph_is_italic:
                                para_text = ' '.join(current_paragraph)
                                markdown_lines.append(f"_{para_text}_")
                                markdown_lines.append("")
                                current_paragraph = []
                            paragraph_is_italic = False
                            current_paragraph.append(line_text.strip())
                    
                    j += 1
            
            else:
                i += 1
        
        # Flush remaining paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            if paragraph_is_italic:
                markdown_lines.append(f"_{para_text}_")
            else:
                markdown_lines.append(para_text)
            markdown_lines.append("")
        
        return markdown_lines

    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str) -> Path:
        """
        Main conversion function
        Output structure:
        output_dir/
          ├── main.md  # <-- File này phải ở đây
          └── images/
              ├── image1.png
              ├── image2.jpg
              └── ...
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Nếu output_dir là "images/", ta lưu main.md ở thư mục cha
        if output_dir.name == "images":
            output_file = output_dir.parent / "main.md"  # <-- Sửa ở đây
        else:
            output_file = output_dir / "main.md"
        
        # Tạo cấu trúc thư mục
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
            
        # Reset global image counter
        self.global_image_counter = 0
        
        doc = fitz.open(pdf_path)
        all_markdown_lines = []
        
        # Thêm tiêu đề file
        file_title = self.get_file_title_from_pdf(str(pdf_path))
        all_markdown_lines.append(file_title)
        all_markdown_lines.append("")
        
        # Xử lý từng page
        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            
            # Extract tất cả elements theo thứ tự
            elements = self.extract_page_elements(page, page_num, images_dir)
            
            # Convert elements sang markdown
            page_markdown = self.extract_styled_text_from_elements(elements)
            all_markdown_lines.extend(page_markdown)
        
        doc.close()
        
        # Cleanup và save
        final_markdown = "\n".join(all_markdown_lines)
        final_markdown = self._cleanup_markdown(final_markdown)
        
        output_file = output_dir / "main.md"
        # Lưu file main.md
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        
        print(f"✅ Conversion completed: {output_file}")
        print(f"   Images saved to: {images_dir}")
        print(f"   Total images: {self.global_image_counter}")
        
        return output_file

    def _cleanup_markdown(self, text: str) -> str:
        """Clean up markdown"""
        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing spaces
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        return text.strip()


from pathlib import Path

if __name__ == "__main__":
    converter = PDFToMarkdownConverter()

    # Thư mục gốc của project (CN-DoubleQ)
    base_dir = Path(__file__).resolve().parent.parent

    # Thư mục chứa input PDF
    input_dir = base_dir / "main" / "data" / "public_test_input" / "public-test-input"
    # input_dir = base_dir / "main" / "data" / "private_test_input" / "private-test-input" 

    # Thư mục output mong muốn
    # base_output_dir = base_dir / "output" / "public_test_output"
    base_output_dir = base_dir / "output" / "private_test_output"
    base_output_dir.mkdir(exist_ok=True, parents=True)

    for pdf_file in input_dir.glob("*.pdf"):
        # Mỗi PDF có thư mục riêng, ví dụ: output/public_test_output/Public080/
        output_dir = base_output_dir / pdf_file.stem / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            converter.convert_pdf_to_markdown(str(pdf_file), str(output_dir))
            print(f"✅ Done: {pdf_file.name} → {output_dir}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()