import os
import re
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

class PDFExtractor:
    def __init__(self):
        # Cấu hình pipeline cho PDF
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def extract_pdf(self, pdf_path, output_dir):
        """
        Trích xuất PDF thành markdown với images và formulas
        
        Args:
            pdf_path: Đường dẫn file PDF
            output_dir: Thư mục output (VD: output/public_test_output/pdf_name/)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"Đang xử lý: {pdf_path}")
        
        # Convert PDF
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        # Export markdown
        markdown_content = doc.export_to_markdown()
        
        # Xử lý images và formulas
        image_counter = 1
        formula_counter = 1
        processed_content = []
        
        # Lưu images từ document
        if hasattr(doc, 'pictures') and doc.pictures:
            for idx, picture in enumerate(doc.pictures, 1):
                try:
                    image_path = images_dir / f"image_{idx}.png"
                    # Lưu ảnh nếu có data
                    if hasattr(picture, 'image') and picture.image:
                        with open(image_path, 'wb') as f:
                            f.write(picture.image.pil_image.tobytes())
                except Exception as e:
                    print(f"Lỗi lưu ảnh {idx}: {e}")
        
        # Replace image references trong markdown
        lines = markdown_content.split('\n')
        for line in lines:
            # Detect images: ![...](...) hoặc <img...>
            if '![' in line or '<img' in line:
                processed_line = re.sub(
                    r'!\[.*?\]\(.*?\)',
                    f'|<image_{image_counter}>|',
                    line
                )
                processed_line = re.sub(
                    r'<img[^>]*>',
                    f'|<image_{image_counter}>|',
                    processed_line
                )
                image_counter += 1
                processed_content.append(processed_line)
            
            # Detect formulas: $ ... $ hoặc $$ ... $$
            elif '$' in line:
                processed_line = re.sub(
                    r'\$\$.*?\$\$',
                    f'|<formula_{formula_counter}>|',
                    line
                )
                processed_line = re.sub(
                    r'\$[^\$]+\$',
                    f'|<formula_{formula_counter}>|',
                    processed_line
                )
                if processed_line != line:
                    formula_counter += 1
                processed_content.append(processed_line)
            else:
                processed_content.append(line)
        
        final_markdown = '\n'.join(processed_content)
        
        # Lưu main.md
        main_md_path = output_dir / "main.md"
        with open(main_md_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        
        print(f"✅ Đã trích xuất: {main_md_path}")
        return final_markdown


def extract_all_pdfs(input_dir, output_base_dir):
    """
    Trích xuất tất cả PDF trong thư mục
    
    Args:
        input_dir: Thư mục chứa các file PDF
        output_base_dir: Thư mục gốc output
    
    Returns:
        dict: {pdf_name: markdown_content}
    """
    extractor = PDFExtractor()
    extracted_data = {}
    
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem  # Tên file không có extension
        output_dir = Path(output_base_dir) / pdf_name
        
        markdown_content = extractor.extract_pdf(str(pdf_path), str(output_dir))
        extracted_data[pdf_name] = markdown_content
    
    return extracted_data


if __name__ == "__main__":
    # Test
    input_dir = "main/data/processed/public_test/pdfs"
    output_dir = "main/output/public_test_output"
    
    extracted_data = extract_all_pdfs(input_dir, output_dir)
    print(f"\n✅ Đã trích xuất {len(extracted_data)} file PDF")