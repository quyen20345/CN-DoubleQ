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
        """Tạo prompt chi tiết cho tác vụ QA trắc nghiệm với kỹ thuật Chain-of-Thought."""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        return f"""Bạn là chuyên gia phân tích tài liệu kỹ thuật. Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm dựa trên tài liệu được cung cấp.

### THÔNG TIN TÀI LIỆU:
{context}

---

### CÂU HỎI:
{question}

### CÁC LỰA CHỌN:
{options_text}

### HƯỚNG DẪN TRẢ LỜI:
1. Đọc kỹ câu hỏi và xác định thông tin cần tìm
2. Tìm kiếm thông tin liên quan trong tài liệu
3. Đối chiếu TỪNG lựa chọn với thông tin đã tìm thấy
4. Lưu ý: Câu hỏi có thể có MỘT hoặc NHIỀU đáp án đúng
5. CHỈ chọn đáp án được XÁC NHẬN RÕ RÀNG bởi tài liệu
6. Nếu không chắc chắn về một đáp án, KHÔNG chọn nó

### YÊU CẦU ĐỊNH DẠNG:
Trả lời ĐÚNG theo format JSON sau (không thêm text nào khác):

{{
  "reasoning": "Giải thích ngắn gọn về cách bạn tìm thấy đáp án",
  "correct_count": <số nguyên>,
  "correct_answers": ["A", "B", ...]
}}

CHÚ Ý: 
- correct_count phải khớp với số phần tử trong correct_answers
- Chỉ trả về JSON, không thêm markdown, backticks hay text giải thích
- Nếu không tìm thấy thông tin rõ ràng, chọn đáp án có khả năng cao nhất

### TRẢ LỜI:
"""

    def _parse_llm_response(self, response: str) -> tuple[int, list]:
        """Phân tích cú pháp phản hồi JSON từ LLM với xử lý fallback tốt hơn."""
        try:
            # Loại bỏ markdown code blocks nếu có
            response = re.sub(r'```json\s*|\s*```', '', response)
            
            # Tìm JSON object
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                answers = sorted([
                    str(ans).upper() 
                    for ans in data.get("correct_answers", []) 
                    if str(ans).upper() in 'ABCD'
                ])
                count = len(answers)

                if count == 0:
                    raise ValueError("Không có đáp án trong JSON.")

                return count, answers

            raise ValueError("Không tìm thấy JSON trong phản hồi.")
            
        except Exception as e:
            print(f"  ⚠ Parse JSON thất bại: {e}. Dùng regex fallback.")
            
            # Fallback 1: Tìm pattern "correct_answers": ["A", "B"]
            pattern = r'["\']correct_answers["\']\s*:\s*\[(.*?)\]'
            match = re.search(pattern, response)
            if match:
                answers_str = match.group(1)
                answers = sorted(list(set(re.findall(r'["\']([A-D])["\']', answers_str))))
                if answers:
                    return len(answers), answers
            
            # Fallback 2: Tìm tất cả chữ cái A-D xuất hiện
            answers = sorted(list(set(re.findall(r'\b([A-D])\b', response))))
            
            # Đảm bảo luôn có ít nhất 1 đáp án
            if not answers:
                # Chiến lược cuối: chọn A nếu không có gì
                print("  ⚠ Không tìm thấy đáp án, mặc định chọn A")
                answers = ["A"]
            
            return len(answers), answers

    def _rerank_results(self, question: str, search_results: list) -> list:
        """Sắp xếp lại kết quả tìm kiếm dựa trên độ liên quan."""
        # Tính điểm dựa trên:
        # 1. Score từ vector search
        # 2. Số lượng từ khóa trùng khớp
        question_words = set(re.findall(r'\w+', question.lower()))
        
        scored_results = []
        for point in search_results:
            content = point.payload.get('content', '').lower()
            content_words = set(re.findall(r'\w+', content))
            
            # Tính keyword overlap
            overlap = len(question_words & content_words)
            keyword_score = overlap / max(len(question_words), 1)
            
            # Kết hợp với vector score
            combined_score = point.score * 0.7 + keyword_score * 0.3
            
            scored_results.append((combined_score, point))
        
        # Sắp xếp theo điểm kết hợp
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [point for _, point in scored_results]

    def _extract_relevant_context(self, question: str, search_results: list, max_tokens: int = 2000) -> str:
        """Trích xuất context liên quan nhất, tránh quá tải token."""
        if not search_results:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        # Rerank kết quả
        ranked_results = self._rerank_results(question, search_results)
        
        context_parts = []
        current_length = 0
        
        for idx, point in enumerate(ranked_results, 1):
            content = point.payload.get('content', '')
            source = point.payload.get('source', 'N/A')
            score = point.score
            
            # Ước lượng độ dài (1 token ~ 4 ký tự)
            estimated_tokens = len(content) // 4
            
            if current_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(
                f"[Nguồn {idx}: {source} | Độ liên quan: {score:.2f}]\n{content}"
            )
            current_length += estimated_tokens
        
        return "\n\n---\n\n".join(context_parts)

    def answer_question(self, question: str, options: dict) -> tuple[int, list]:
        """Trả lời câu hỏi với pipeline RAG được tối ưu."""
        # Làm sạch options - chuyển tất cả về string và xử lý NaN
        cleaned_options = {}
        for key, value in options.items():
            if pd.isna(value):
                cleaned_options[key] = ""
            else:
                cleaned_options[key] = str(value).strip()
        
        # Bước 1: Tìm kiếm với multiple queries
        queries = [
            question,  # Câu hỏi gốc
            f"{question} {' '.join(v for v in cleaned_options.values() if v)}"  # Câu hỏi + options có giá trị
        ]
        
        all_results = []
        for query in queries:
            results = self.vector_store.search(query, top_k=3, threshold=0.25)
            all_results.extend(results)
        
        # Loại bỏ trùng lặp (dựa trên content)
        seen_contents = set()
        unique_results = []
        for point in all_results:
            content = point.payload.get('content', '')
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(point)
        
        # Bước 2: Trích xuất context tốt nhất
        context = self._extract_relevant_context(question, unique_results, max_tokens=2000)
        
        # Bước 3: Tạo prompt và gọi LLM (dùng cleaned_options)
        prompt = self._create_qa_prompt(question, cleaned_options, context)
        response = self.llm.invoke(prompt)
        
        # Bước 4: Parse kết quả
        count, answers = self._parse_llm_response(response)

        # Đảm bảo luôn có ít nhất 1 đáp án
        if count == 0 or not answers:
            print("  ⚠ Không có đáp án hợp lệ, mặc định chọn A")
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
            
            print(f"\n{'='*60}")
            print(f"Câu {idx + 1}/{total}: {str(question)[:80]}...")
            print(f"{'='*60}")
            
            count, answers = self.answer_question(question, options)
            results.append((count, answers))
            
            print(f"✅ Kết quả: {count} đáp án đúng → {', '.join(answers)}")
        
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
                if not answers:
                    count, answers = 1, ["A"]

                if len(answers) > 1:
                    answers_str = f"\"{','.join(answers)}\""
                else:
                    answers_str = answers[0]

                f.write(f"{count},{answers_str}\n")

        print("✅ Đã tạo file answer.md thành công.")

    def create_zip(self, zip_name: str):
        """Tạo file .zip theo cấu trúc chuẩn."""
        project_root = self.output_dir.parent
        zip_path = project_root / zip_name

        print(f"\n📦 Đang nén '{self.output_dir}' thành '{zip_path}'...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.answer_md_path, arcname="answer.md")

            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path(self.output_dir.name) / file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname=arcname)

        print(f"✅ Đã tạo file zip thành công tại: {zip_path}")