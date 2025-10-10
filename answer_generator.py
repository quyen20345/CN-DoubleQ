from pathlib import Path

class AnswerGenerator:
    """
    Tạo file answer.md theo format yêu cầu:
    
    ### TASK EXTRACT
    # pdf_name_1
    [nội dung main.md]
    
    # pdf_name_2
    [nội dung main.md]
    
    ### TASK QA
    [số câu đúng]
    [đáp án]
    [số câu đúng]
    [đáp án]
    ...
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
    
    def generate_answer_md(self, extracted_data, qa_results):
        """
        Tạo file answer.md
        
        Args:
            extracted_data: dict {pdf_name: markdown_content}
            qa_results: list [(correct_count, correct_answers), ...]
        """
        answer_content = []
        
        # PHẦN 1: TASK EXTRACT
        answer_content.append("### TASK EXTRACT\n")
        
        for pdf_name, markdown_content in extracted_data.items():
            answer_content.append(f"# {pdf_name}\n")
            answer_content.append(markdown_content)
            answer_content.append("\n")
        
        # PHẦN 2: TASK QA
        answer_content.append("### TASK QA\n")
        
        for correct_count, correct_answers in qa_results:
            answer_content.append(f"{correct_count}\n")
            answer_content.append(f"{','.join(correct_answers)}\n")
        
        # Ghi file
        answer_md_path = self.output_dir / "answer.md"
        with open(answer_md_path, 'w', encoding='utf-8') as f:
            f.write(''.join(answer_content))
        
        print(f"\n✅ Đã tạo file: {answer_md_path}")
        return answer_md_path
    
    def copy_main_py(self, main_py_source):
        """
        Copy file main.py vào thư mục output
        
        Args:
            main_py_source: Đường dẫn file main.py nguồn
        """
        import shutil
        
        dest = self.output_dir / "main.py"
        shutil.copy(main_py_source, dest)
        print(f"✅ Đã copy main.py vào: {dest}")
    
    def create_zip(self, zip_name="public_test_output.zip"):
        """
        Tạo file zip từ thư mục output
        
        Args:
            zip_name: Tên file zip
        """
        import shutil
        
        zip_path = self.output_dir.parent / zip_name
        
        # Xóa file zip cũ nếu có
        if zip_path.exists():
            zip_path.unlink()
        
        # Tạo zip
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            self.output_dir
        )
        
        print(f"✅ Đã tạo file: {zip_path}")
        return zip_path


def create_complete_output(extracted_data, qa_results, output_dir, main_py_path=None):
    """
    Tạo output hoàn chỉnh bao gồm:
    - answer.md
    - main.py
    - Các thư mục PDF đã trích xuất
    
    Args:
        extracted_data: dict {pdf_name: markdown_content}
        qa_results: list [(correct_count, correct_answers), ...]
        output_dir: Thư mục output
        main_py_path: Đường dẫn file main.py để copy (optional)
    """
    generator = AnswerGenerator(output_dir)
    
    # Tạo answer.md
    generator.generate_answer_md(extracted_data, qa_results)
    
    # Copy main.py nếu có
    if main_py_path and Path(main_py_path).exists():
        generator.copy_main_py(main_py_path)
    else:
        print("⚠️ Không tìm thấy main.py để copy")
    
    return generator


if __name__ == "__main__":
    # Test
    extracted_data = {
        "sample_pdf": "# Sample Content\n\nThis is a test."
    }
    
    qa_results = [
        (2, ['A', 'C']),
        (1, ['B']),
        (3, ['A', 'B', 'D'])
    ]
    
    output_dir = "main/output/public_test_output"
    
    generator = create_complete_output(
        extracted_data,
        qa_results,
        output_dir
    )
    
    # Tạo zip
    generator.create_zip("public_test_output.zip")