# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
from pathlib import Path
import shutil
import zipfile

from main.src.llm.llm_integrations import get_llm
from main.src.vectordb.qdrant import VectorStore


class QAHandler:
    """Xử lý toàn bộ logic cho việc trả lời câu hỏi."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = get_llm()

    def _create_qa_prompt(self, question: str, options: dict, context: str) -> str:
        """Tạo prompt chi tiết cho tác vụ QA trắc nghiệm."""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        return f"""Bạn là một chuyên gia phân tích tài liệu kỹ thuật. Dựa vào "THÔNG TIN TÀI LIỆU" dưới đây để trả lời câu hỏi trắc nghiệm một cách chính xác.

### THÔNG TIN TÀI LIỆU:
{context}

---

### CÂU HỎI:
{question}

### CÁC LỰA CHỌN:
{options_text}

### YÊU CẦU:
1. Đọc kỹ câu hỏi và tất cả các lựa chọn.
2. Đối chiếu TỪNG lựa chọn với "THÔNG TIN TÀI LIỆU".
3. Câu hỏi có thể có MỘT hoặc NHIỀU đáp án đúng.
4. Chỉ chọn những đáp án được xác nhận HOÀN TOÀN bởi tài liệu.
5. Trả lời theo định dạng JSON nghiêm ngặt sau đây, không thêm bất kỳ giải thích nào khác.

{{
  "correct_count": <số lượng đáp án đúng>,
  "correct_answers": ["<A>", "<B>", ...]
}}

### TRẢ LỜI (CHỈ JSON):
"""

    def _parse_llm_response(self, response: str) -> tuple[int, list]:
        """Phân tích cú pháp phản hồi JSON từ LLM, đảm bảo luôn có ít nhất 1 đáp án."""
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                answers = sorted([str(ans).upper() for ans in data.get("correct_answers", []) if str(ans).upper() in 'ABCD'])
                count = len(answers)

                # Nếu không có đáp án nào, fallback sang regex
                if count == 0:
                    raise ValueError("Không có đáp án trong JSON.")

                if count != data.get("correct_count", 0):
                    print(f"  > Cảnh báo: Số lượng đáp án không khớp. Tự động sửa lại.")
                return count, answers

            raise ValueError("Không tìm thấy JSON trong phản hồi.")
        except Exception as e:
            print(f"  > Cảnh báo: Lỗi khi parse LLM JSON ({e}). Fallback sang regex.")
            answers = sorted(list(set(re.findall(r'\b([A-D])\b', response.upper()))))
            
            # ✅ Đảm bảo luôn có ít nhất 1 đáp án
            if not answers:
                answers = ["A"]
            return len(answers), answers

    def answer_question(self, question: str, options: dict) -> tuple[int, list]:
        """Tìm kiếm context và trả lời một câu hỏi."""
        search_results = self.vector_store.search(question, top_k=5, threshold=0.3)
        
        context = "\n\n---\n\n".join([
            f"Nguồn: {point.payload.get('source', 'N/A')}\n\n{point.payload.get('content', '')}"
            for point in search_results
        ]) if search_results else "Không có thông tin nào được tìm thấy trong tài liệu."
        
        prompt = self._create_qa_prompt(question, options, context)
        response = self.llm.invoke(prompt)
        count, answers = self._parse_llm_response(response)

        # ✅ Bảo đảm luôn có ít nhất 1 đáp án khi ghi file
        if count == 0 or not answers:
            print("  > Không có đáp án hợp lệ, tự động gán 'A'")
            count, answers = 1, ["A"]

        return count, answers

    def process_questions_csv(self, csv_path: Path) -> list[tuple] | None:
        """Xử lý file CSV chứa các câu hỏi."""
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy file question.csv tại '{csv_path}'")
            return None
            
        results = []
        total = len(df)
        print(f"\n🤔 Bắt đầu trả lời {total} câu hỏi...")
        
        for idx, row in df.iterrows():
            question = row.iloc[0]
            options = { 'A': row.iloc[1], 'B': row.iloc[2], 'C': row.iloc[3], 'D': row.iloc[4] }
            
            print(f"\nCâu {idx + 1}/{total}: {str(question)[:80]}...")
            
            count, answers = self.answer_question(question, options)
            results.append((count, answers))
            
            print(f"  ➜ Kết quả: {count} câu đúng - Đáp án: {', '.join(answers)}")
        
        return results


class AnswerGenerator:
    """Tạo file answer.md và file .zip để nộp bài."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.answer_md_path = self.output_dir / "answer.md"

    def generate_answer_md(self, extracted_data: dict, qa_results: list):
        """Tạo nội dung file answer.md tổng hợp theo định dạng chuẩn yêu cầu."""
        print(f"\n📝 Đang tạo file kết quả tại: {self.answer_md_path}")

        with self.answer_md_path.open("w", encoding="utf-8") as f:
            # --- Phần 1: TASK EXTRACT ---
            f.write("### TASK EXTRACT\n")
            for pdf_name in sorted(extracted_data.keys()):
                pdf_title = Path(pdf_name).stem
                f.write(f"# {pdf_title}\n\n")
                f.write(extracted_data[pdf_name].strip() + "\n\n")

            # --- Phần 2: TASK QA ---
            f.write("### TASK QA\n")
            f.write("num_correct,answers\n")
            for count, answers in qa_results:
                if not answers:  # ✅ đảm bảo không rỗng
                    count, answers = 1, ["A"]

                # ✅ Nếu có nhiều đáp án, dùng ngoặc kép "A,B"
                if len(answers) > 1:
                    answers_str = f"\"{','.join(answers)}\""
                else:
                    answers_str = answers[0]

                f.write(f"{count},{answers_str}\n")

        print("✅ Đã tạo file answer.md thành công.")

    def create_zip(self, zip_name: str):
        """Tạo file .zip theo cấu trúc chuẩn:
        zip_name.zip
        ├── answer.md
        └── output_dir/
            ├── main.py
            ├── Publicxxx/
            └── ...
        """
        project_root = self.output_dir.parent
        zip_path = project_root / zip_name

        print(f"\n📦 Đang nén '{self.output_dir}' thành '{zip_path}'...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 1️⃣ Thêm file answer.md ở ngoài cùng
            zipf.write(self.answer_md_path, arcname="answer.md")

            # 2️⃣ Thêm toàn bộ nội dung trong output_dir (public_test_output)
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path(self.output_dir.name) / file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname=arcname)

        print(f"✅ Đã tạo file zip thành công tại: {zip_path}")