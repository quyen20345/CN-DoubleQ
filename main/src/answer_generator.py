# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
from pathlib import Path

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
        """Phân tích cú pháp phản hồi JSON từ LLM, có xử lý lỗi."""
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                answers = sorted([str(ans).upper() for ans in data.get("correct_answers", []) if str(ans).upper() in 'ABCD'])
                count = len(answers)
                # Ghi đè count từ LLM để đảm bảo tính nhất quán
                if count != data.get("correct_count", 0):
                     print(f"  > Cảnh báo: Số lượng đáp án không khớp. Tự động sửa lại.")
                return count, answers
            raise ValueError("Không tìm thấy JSON trong response.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  > Cảnh báo: Không thể parse JSON từ LLM. Lỗi: {e}. Fallback sang regex.")
            answers = sorted(list(set(re.findall(r'\b([A-D])\b', response.upper()))))
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
        return self._parse_llm_response(response)

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
            
            print(f"  ➜ Kết quả: {count} câu đúng - Đáp án: {', '.join(answers) if answers else 'Không có'}")
        
        return results
