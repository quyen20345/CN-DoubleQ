# -*- coding: utf-8 -*-
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from main.src.llm.llm_integrations import get_llm
from main.src.vectordb.qdrant import VectorStore


class QAHandler:
    """Xử lý toàn bộ logic cho việc trả lời câu hỏi với độ chính xác cao hơn."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = get_llm()

    def _create_qa_prompt(self, question: str, options: dict, context: str) -> str:
        """Tạo prompt tối ưu với Chain-of-Thought và few-shot examples."""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        return f"""Bạn là chuyên gia phân tích tài liệu kỹ thuật IoT/Smart Home với khả năng reasoning cao.

### NGUYÊN TẮC QUAN TRỌNG:
1. CHỈ chọn đáp án được KHẲNG ĐỊNH RÕ RÀNG trong tài liệu
2. Nếu tài liệu KHÔNG ĐỀ CẬP, đáp án đó là SAI
3. Câu hỏi có thể có 1, 2, 3, hoặc 4 đáp án đúng
4. Đọc KỸ từng lựa chọn, không bỏ sót chi tiết
5. Chú ý từ phủ định: "KHÔNG", "NGOẠI TRỪ", "TRỪ"

### TÀI LIỆU THAM KHẢO:
{context}

---

### CÂU HỎI:
{question}

### CÁC LỰA CHỌN:
{options_text}

### PHƯƠNG PHÁP TRẢ LỜI (THỰC HIỆN TUẦN TỰ):

**Bước 1: Phân tích câu hỏi**
- Xác định thông tin cần tìm
- Chú ý từ khóa quan trọng
- Phát hiện câu hỏi phủ định (nếu có)

**Bước 2: Tìm bằng chứng trong tài liệu**
- Duyệt qua tài liệu tìm thông tin liên quan
- Ghi chú nguồn (Đoạn số mấy)

**Bước 3: Đối chiếu TỪNG lựa chọn**
- A: [Có trong tài liệu? → Đúng/Sai vì...]
- B: [Có trong tài liệu? → Đúng/Sai vì...]
- C: [Có trong tài liệu? → Đúng/Sai vì...]
- D: [Có trong tài liệu? → Đúng/Sai vì...]

**Bước 4: Kết luận**
- Liệt kê TẤT CẢ đáp án đúng
- Kiểm tra lại có bỏ sót không

### YÊU CẦU ĐỊNH DẠNG (BỮA BUỘC):
Trả lời ĐÚNG format JSON (không thêm markdown hay text nào khác):

{{
  "reasoning": "Giải thích ngắn gọn cách tìm đáp án và lý do chọn",
  "analysis": {{
    "A": "Đúng/Sai - Lý do",
    "B": "Đúng/Sai - Lý do",
    "C": "Đúng/Sai - Lý do",
    "D": "Đúng/Sai - Lý do"
  }},
  "correct_count": <số nguyên từ 1-4>,
  "correct_answers": ["A", "B", ...]
}}

### LƯU Ý QUAN TRỌNG:
- Nếu câu hỏi dạng "Điều nào SAI?", chọn đáp án KHÔNG đúng với tài liệu
- Nếu không chắc chắn 100%, KHÔNG chọn đáp án đó
- correct_count PHẢI khớp với số phần tử trong correct_answers
- Luôn có ít nhất 1 đáp án đúng

### TRẢ LỜI:
"""

    def _parse_llm_response(self, response: str) -> Tuple[int, List[str]]:
        """Parse JSON với fallback thông minh hơn."""
        try:
            # Loại bỏ markdown
            response = re.sub(r'```json\s*|\s*```', '', response.strip())
            
            # Tìm JSON object
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                answers = sorted([
                    str(ans).upper() 
                    for ans in data.get("correct_answers", []) 
                    if str(ans).upper() in 'ABCD'
                ])
                
                if not answers:
                    raise ValueError("Không có đáp án trong JSON")
                
                # Validate count
                count = len(answers)
                declared_count = data.get("correct_count", count)
                
                if count != declared_count:
                    print(f"  ⚠ Cảnh báo: count không khớp ({declared_count} vs {count}), dùng {count}")
                
                return count, answers
            
            raise ValueError("Không tìm thấy JSON")
            
        except Exception as e:
            print(f"  ⚠ Parse thất bại: {e}. Dùng fallback.")
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Tuple[int, List[str]]:
        """Fallback parsing với nhiều chiến lược."""
        # Strategy 1: Tìm "correct_answers": [...]
        pattern1 = r'["\']correct_answers["\']\s*:\s*\[(.*?)\]'
        match = re.search(pattern1, response, re.DOTALL)
        if match:
            answers_str = match.group(1)
            answers = sorted(list(set(re.findall(r'["\']([A-D])["\']', answers_str))))
            if answers:
                return len(answers), answers
        
        # Strategy 2: Tìm pattern "A, B, C"
        pattern2 = r'(?:đáp án|answers?)[\s:]+([A-D](?:\s*,\s*[A-D])*)'
        match = re.search(pattern2, response, re.IGNORECASE)
        if match:
            answers = sorted(list(set(re.findall(r'[A-D]', match.group(1)))))
            if answers:
                return len(answers), answers
        
        # Strategy 3: Đếm số lần xuất hiện của mỗi chữ cái
        counts = {letter: len(re.findall(rf'\b{letter}\b', response)) 
                  for letter in 'ABCD'}
        
        # Chọn các chữ cái xuất hiện nhiều nhất (threshold = 2)
        candidates = [k for k, v in counts.items() if v >= 2]
        if candidates:
            return len(candidates), sorted(candidates)
        
        # Strategy 4: Lấy tất cả A-D xuất hiện
        all_letters = re.findall(r'\b([A-D])\b', response)
        if all_letters:
            unique = sorted(list(set(all_letters)))
            # Nếu quá nhiều (>2), chỉ lấy 2 đầu
            if len(unique) > 2:
                unique = unique[:2]
            return len(unique), unique
        
        # Final fallback: chọn A
        print("  ⚠ Không parse được, mặc định chọn A")
        return 1, ["A"]

    def _expand_query(self, question: str, options: dict) -> List[str]:
        """Tạo nhiều query variants để tăng khả năng tìm thấy thông tin."""
        queries = [question]  # Query gốc
        
        # Thêm query với keywords từ options
        valid_options = [v for v in options.values() if v and str(v).strip()]
        if valid_options:
            # Lấy 2-3 từ khóa quan trọng từ mỗi option
            keywords = []
            for opt in valid_options[:2]:  # Chỉ lấy 2 options đầu
                words = re.findall(r'\b\w{4,}\b', str(opt))
                keywords.extend(words[:3])
            
            if keywords:
                queries.append(f"{question} {' '.join(keywords[:5])}")
        
        # Trích xuất keywords từ câu hỏi
        question_keywords = re.findall(r'\b\w{4,}\b', question)
        if len(question_keywords) > 3:
            queries.append(" ".join(question_keywords[:7]))
        
        return queries

    def _rerank_with_keywords(self, question: str, results: list, options: dict) -> list:
        """Re-rank kết quả dựa trên keyword matching."""
        # Extract keywords từ question và options
        all_text = question + " " + " ".join(str(v) for v in options.values() if v)
        keywords = set(re.findall(r'\b\w{3,}\b', all_text.lower()))
        
        scored = []
        for point in results:
            content = point.payload.get('content', '').lower()
            
            # Tính keyword overlap
            content_words = set(re.findall(r'\b\w{3,}\b', content))
            overlap = len(keywords & content_words)
            keyword_score = overlap / max(len(keywords), 1)
            
            # Tính density (keyword xuất hiện gần nhau hơn = tốt hơn)
            density_score = self._calculate_keyword_density(content, keywords)
            
            # Kết hợp điểm
            combined = point.score * 0.5 + keyword_score * 0.3 + density_score * 0.2
            scored.append((combined, point))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def _calculate_keyword_density(self, text: str, keywords: set) -> float:
        """Tính mật độ keywords (keywords xuất hiện gần nhau)."""
        positions = []
        words = text.split()
        
        for i, word in enumerate(words):
            if word.lower() in keywords:
                positions.append(i)
        
        if len(positions) < 2:
            return 0.0
        
        # Tính khoảng cách trung bình giữa các keywords
        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_distance = sum(distances) / len(distances)
        
        # Score cao hơn khi keywords gần nhau
        return 1.0 / (1.0 + avg_distance / 10)

    def _extract_context_smart(self, question: str, results: list, 
                               options: dict, max_tokens: int = 2500) -> str:
        """Trích xuất context thông minh với prioritization."""
        if not results:
            return "Không tìm thấy thông tin liên quan."
        
        # Re-rank
        ranked = self._rerank_with_keywords(question, results, options)
        
        context_parts = []
        current_tokens = 0
        seen_content = set()
        
        for idx, point in enumerate(ranked, 1):
            content = point.payload.get('content', '').strip()
            source = point.payload.get('source', 'N/A')
            score = point.score
            
            # Skip duplicates
            content_hash = hash(content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Estimate tokens
            est_tokens = len(content) // 4
            
            if current_tokens + est_tokens > max_tokens:
                break
            
            # Highlight keywords (optional, giúp LLM focus)
            highlighted = self._highlight_keywords(content, question, options)
            
            context_parts.append(
                f"[Đoạn {idx} - Nguồn: {source} | Score: {score:.3f}]\n{highlighted}"
            )
            current_tokens += est_tokens
        
        return "\n\n" + "="*60 + "\n\n".join(context_parts)

    def _highlight_keywords(self, text: str, question: str, options: dict) -> str:
        """Highlight keywords quan trọng bằng ** **."""
        # Extract keywords
        all_text = question + " " + " ".join(str(v) for v in options.values() if v)
        keywords = set(re.findall(r'\b\w{4,}\b', all_text.lower()))
        
        # Highlight (chỉ highlight 1 lần để không làm rối)
        highlighted = text
        for kw in keywords:
            pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
            # Chỉ thay thế lần đầu tiên
            highlighted = pattern.sub(r'**\1**', highlighted, count=1)
        
        return highlighted

    def answer_question(self, question: str, options: dict) -> Tuple[int, List[str]]:
        """Pipeline RAG được tối ưu hóa cao."""
        # Clean options
        cleaned_options = {}
        for key, value in options.items():
            if pd.isna(value):
                cleaned_options[key] = ""
            else:
                cleaned_options[key] = str(value).strip()
        
        # Bước 1: Multi-query search
        queries = self._expand_query(question, cleaned_options)
        
        all_results = []
        for query in queries:
            # Tăng top_k và giảm threshold để recall cao hơn
            results = self.vector_store.search(query, top_k=5, threshold=0.2)
            all_results.extend(results)
        
        # Deduplicate
        seen = set()
        unique_results = []
        for point in all_results:
            content = point.payload.get('content', '')
            if content not in seen:
                seen.add(content)
                unique_results.append(point)
        
        # Bước 2: Extract context thông minh
        context = self._extract_context_smart(
            question, unique_results, cleaned_options, max_tokens=2500
        )
        
        # Bước 3: Generate prompt và gọi LLM
        prompt = self._create_qa_prompt(question, cleaned_options, context)
        response = self.llm.invoke(prompt)
        
        # Bước 4: Parse kết quả
        count, answers = self._parse_llm_response(response)
        
        # Validation cuối
        if count == 0 or not answers:
            print("  ⚠ Fallback: chọn A")
            count, answers = 1, ["A"]
        
        return count, answers

    def process_questions_csv(self, csv_path: Path) -> List[Tuple] | None:
        """Xử lý CSV với progress tracking."""
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ Không tìm thấy {csv_path}")
            return None
        
        results = []
        total = len(df)
        print(f"\n🤔 Bắt đầu trả lời {total} câu hỏi...\n")
        
        for idx, row in df.iterrows():
            question = row.iloc[0]
            options = {
                'A': row.iloc[1], 
                'B': row.iloc[2], 
                'C': row.iloc[3], 
                'D': row.iloc[4]
            }
            
            print(f"\n{'='*70}")
            print(f"Câu {idx + 1}/{total}: {str(question)[:100]}...")
            print(f"{'='*70}")
            
            count, answers = self.answer_question(question, options)
            results.append((count, answers))
            
            print(f"✅ Kết quả: {count} đáp án → {', '.join(answers)}")
            print(f"Progress: [{idx + 1}/{total}] ({(idx + 1) / total * 100:.1f}%)")
        
        return results