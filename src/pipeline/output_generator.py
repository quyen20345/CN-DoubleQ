# src/pipeline/output_generator.py
"""
Module này chịu trách nhiệm tạo ra các file output cuối cùng theo định dạng
yêu cầu của cuộc thi, bao gồm file `answer.md` và file `.zip` nén toàn bộ kết quả.
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

class OutputGenerator:
    """
    Tạo các file output cuối cùng (answer.md, .zip).
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.answer_md_path = self.output_dir / "answer.md"

    def _generate_answer_md(self, extracted_data: Dict, qa_results: List[Tuple]):
        """Tạo file answer.md với định dạng chuẩn."""
        print(f"\n📝 Đang tạo file: {self.answer_md_path}...")

        with self.answer_md_path.open("w", encoding="utf-8") as f:
            # Phần TASK EXTRACT
            f.write("### TASK EXTRACT\n")
            # Sắp xếp theo tên để đảm bảo thứ tự nhất quán
            for pdf_name in sorted(extracted_data.keys()):
                f.write(extracted_data[pdf_name].strip() + "\n\n")

            # Phần TASK QA
            f.write("### TASK QA\n")
            f.write("num_correct,answers\n")
            for count, answers in qa_results:
                # Định dạng câu trả lời có nhiều đáp án trong dấu ngoặc kép
                answers_str = f'"{",".join(answers)}"' if len(answers) > 1 else answers[0]
                f.write(f"{count},{answers_str}\n")
        
        print("✅ Tạo answer.md thành công.")

    def _create_zip_archive(self, zip_name: str):
        """Tạo file .zip chứa toàn bộ thư mục output."""
        # Đường dẫn file zip sẽ nằm ngoài thư mục output
        zip_path = self.output_dir.parent / zip_name
        print(f"\n📦 Đang nén kết quả vào: {zip_path}...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Duyệt qua tất cả các file trong thư mục output
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    # Tạo đường dẫn tương đối để giữ cấu trúc thư mục trong zip
                    arcname = file_path.relative_to(self.output_dir.parent)
                    zipf.write(file_path, arcname=arcname)
        
        print(f"✅ Nén file zip thành công.")
        
    def generate_final_output(self, extracted_data: Dict, qa_results: List[Tuple], zip_name: str):
        """
        Hàm chính điều phối việc tạo tất cả các file output.
        """
        self._generate_answer_md(extracted_data, qa_results)
        self._create_zip_archive(zip_name)
