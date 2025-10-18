# src/rag_system/qa_handler.py
"""
Module này chứa class QAHandler, chịu trách nhiệm xử lý toàn bộ logic
để trả lời câu hỏi, từ việc lấy context, tạo prompt, gọi LLM và phân tích kết quả.
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from src.llm.client import get_llm
from .retriever import HybridRetriever # <-- THAY ĐỔI: Import HybridRetriever

class QAHandler:
    """
    Xử lý logic trả lời câu hỏi bằng cách sử dụng một retriever.
    """
    
    def __init__(self, retriever: HybridRetriever): # <-- THAY ĐỔI: Sử dụng HybridRetriever
        self.retriever = retriever
        self.llm = get_llm()

    def _create_qa_prompt(self, question: str, options: dict, context: str) -> str:
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        return f"""Bạn là chuyên gia phân tích tài liệu kỹ thuật IoT/Smart Home với khả năng reasoning cao.

### NGUYÊN TẮC QUAN TRỌNG:
1. CHỈ chọn đáp án được KHẲNG ĐỊNH RÕ RÀNG trong tài liệu.
2. Nếu tài liệu KHÔNG ĐỀ CẬP hoặc KHÔNG ĐỦ BẰNG CHỨNG, đáp án đó là SAI.
3. Câu hỏi có thể có MỘT hoặc NHIỀU đáp án đúng.
4. Đọc KỸ từng lựa chọn, không bỏ sót chi tiết.

### TÀI LIỆU THAM KHẢO:
{context}

---

### CÂU HỎI:
{question}

### CÁC LỰA CHỌN:
{options_text}

### YÊU CẦU ĐỊNH DẠNG (BẮT BUỘC):
Trả lời ĐÚNG format JSON (không thêm markdown hay text nào khác):

{{
  "reasoning": "Giải thích ngắn gọn từng bước suy luận, đối chiếu từng lựa chọn với tài liệu tham khảo.",
  "analysis": {{
    "A": "Đúng/Sai - Lý do",
    "B": "Đúng/Sai - Lý do",
    "C": "Đúng/Sai - Lý do",
    "D": "Đúng/Sai - Lý do"
  }},
  "correct_count": <số nguyên từ 1-4>,
  "correct_answers": ["A", "B", ...]
}}

### TRẢ LỜI:
"""

    def _parse_llm_response(self, response: str) -> Tuple[int, List[str]]:
        try:
            response = re.sub(r'```json\s*|\s*```', '', response.strip())
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                answers = sorted([str(ans).upper() for ans in data.get("correct_answers", []) if str(ans).upper() in 'ABCD'])
                if not answers: raise ValueError("Không có đáp án trong JSON")
                
                count = len(answers)
                declared_count = data.get("correct_count", count)
                if count != declared_count:
                    print(f"  ⚠ Cảnh báo: count không khớp ({declared_count} vs {count}), dùng {count}")
                return count, answers
            
            raise ValueError("Không tìm thấy JSON")
            
        except Exception as e:
            print(f"  ⚠ Parse thất bại: {e}. Dùng fallback.")
            # Fallback đơn giản: tìm các chữ cái A,B,C,D trong response
            found_answers = sorted(list(set(re.findall(r'\b([A-D])\b', response.upper()))))
            if found_answers:
                return len(found_answers), found_answers
            return 1, ["A"] # Fallback cuối cùng

    def _format_context(self, documents: List[Dict[str, str]]) -> str:
        """Định dạng context từ các tài liệu được truy xuất."""
        if not documents:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[Đoạn {i+1} - Nguồn: {doc['source']}]\n{doc['content']}")
        
        return "\n\n" + "="*40 + "\n\n".join(context_parts)

    def answer_question(self, question: str, options: dict) -> Tuple[int, List[str]]:
        """
        Pipeline RAG hoàn chỉnh cho một câu hỏi.
        """
        cleaned_options = {k: str(v).strip() if pd.notna(v) else "" for k, v in options.items()}
        
        # Bước 1: Truy xuất tài liệu bằng Hybrid Retriever
        retrieved_docs = self.retriever.retrieve(question, top_k=10)
        
        # Bước 2: Tạo context
        context = self._format_context(retrieved_docs)
        
        # Bước 3: Generate prompt và gọi LLM
        prompt = self._create_qa_prompt(question, cleaned_options, context)
        response = self.llm.invoke(prompt)
        
        # Bước 4: Parse kết quả
        return self._parse_llm_response(response)

    def process_questions_csv(self, csv_path: Path) -> List[Tuple] | None:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file câu hỏi: {csv_path}")
            return None
        
        results = []
        total = len(df)
        print(f"\n🤔 Bắt đầu trả lời {total} câu hỏi...\n")
        
        for idx, row in df.iterrows():
            question = row.iloc[0]
            options = {'A': row.iloc[1], 'B': row.iloc[2], 'C': row.iloc[3], 'D': row.iloc[4]}
            
            print(f"\n{'='*70}\nCâu {idx + 1}/{total}: {str(question)[:100]}...\n{'='*70}")
            
            count, answers = self.answer_question(question, options)
            results.append((count, answers))
            
            print(f"✅ Kết quả: {count} đáp án → {', '.join(answers)}")
            print(f"Progress: [{idx + 1}/{total}] ({(idx + 1) / total * 100:.1f}%)")
        
        return results


    def test_rag_qa(self, question: str) -> str:
        """
        Hàm dùng để test kết quả retrieval
        input: question
        output: top_k retrieval
        """
        print(f"Testing retrieval for question: {question[:100]}")
        
        res = self.retriever.retrieve(question, top_k=5)
        
        if not res:
            return "Result isn't suitable."

        print(f"Found {len(res)} results.\n")

        for i, result in enumerate(res):
            content = result.get('content', 'N/A')
            source = result.get('source', 'N/A')
            score = result.get('score', 0)
            print(f"[{i+1}] Source: {source}")
            print(f"    Score: {score:.3f}")
            print(f"    Content: {content[:200]}...\n")
        return f"Retrieval {len(res)} results."