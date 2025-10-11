# -*- coding: utf-8 -*-
import re
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
except ImportError:
    print("Lỗi: docling chưa được cài. Vui lòng chạy: pip install docling")
    sys.exit(1)

class PDFExtractor:
    """Trích xuất nội dung từ tệp PDF sang Markdown."""
    def __init__(self):
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except Exception as e:
            print(f"Lỗi khi khởi tạo PDFExtractor: {e}")
            sys.exit(1)
    
    def extract_pdf(self, pdf_path: Path, output_dir: Path) -> str:
        """Trích xuất một file PDF duy nhất."""
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"Đang xử lý PDF: {pdf_path.name}")
        try:
            result = self.converter.convert(str(pdf_path))
            doc = result.document
            
            md_content = doc.export_to_markdown()
            # Xóa các link ảnh base64 để thay bằng placeholder
            md_content = re.sub(r'!\[.*?\]\(data:image/.*?\)', r'|<image_placeholder>|', md_content)
            md_content = re.sub(r'<img src="data:image/.*?">', r'|<image_placeholder>|', md_content)
            
            # Đánh số lại các placeholder
            image_counter, formula_counter = 1, 1
            
            def replace_placeholder(prefix):
                def inner_replace(match):
                    nonlocal image_counter, formula_counter
                    if prefix == "image":
                        res = f"|<{prefix}_{image_counter}>|"
                        image_counter += 1
                    else:
                        res = f"|<{prefix}_{formula_counter}>|"
                        formula_counter += 1
                    return res
                return inner_replace
                
            md_content = re.sub(r'\|<image_placeholder>\|', replace_placeholder("image"), md_content)
            md_content = re.sub(r'\$\$[\s\S]*?\$\$', replace_placeholder("formula"), md_content)
            md_content = re.sub(r'\$[^\$]*?\$', replace_placeholder("formula"), md_content)

            main_md_path = output_dir / "main.md"
            main_md_path.write_text(md_content, encoding='utf-8')
            
            print(f"✅ Đã trích xuất xong: {main_md_path}")
            return md_content
        except Exception as e:
            print(f"❌ Lỗi khi trích xuất file {pdf_path.name}: {e}")
            return ""

    def extract_all_pdfs(self, input_dir: Path, output_base_dir: Path) -> dict:
        """Trích xuất tất cả các file PDF trong một thư mục."""
        extracted_data = {}
        pdf_files = list(input_dir.rglob("*.pdf"))
        
        if not pdf_files:
            print(f"Cảnh báo: Không tìm thấy file PDF nào trong '{input_dir}'")
            return {}

        for pdf_path in pdf_files:
            pdf_name = pdf_path.stem
            output_dir = output_base_dir / pdf_name
            markdown_content = self.extract_pdf(pdf_path, output_dir)
            if markdown_content:
                extracted_data[pdf_name] = markdown_content
        return extracted_data
