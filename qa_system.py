import os
import pandas as pd
from pathlib import Path
from main.src.llm.chat import ask_llm
from main.src.utils.collections import COLLECTIONS
from main.src.utils.indexer import load_and_index_data
from main.src.utils._utils import chunking

class QASystem:
    def __init__(self, collection_name="technical_docs"):
        """
        Khởi tạo hệ thống QA
        
        Args:
            collection_name: Tên collection trong vector database
        """
        from main.src.vectordb.qdrant import VectorStore
        from main.src.embedding.model import DenseEmbedding
        
        dense_model = DenseEmbedding()
        self.vector_store = VectorStore(
            collection_name=collection_name,
            dense_model=dense_model
        )
    
    def index_extracted_data(self, extracted_data):
        """
        Index dữ liệu đã trích xuất vào vector database
        
        Args:
            extracted_data: dict {pdf_name: markdown_content}
        """
        print("🔄 Đang index dữ liệu...")
        self.vector_store.recreate_collection()
        
        all_chunks = []
        for pdf_name, content in extracted_data.items():
            chunks = chunking(content)
            for chunk in chunks:
                all_chunks.append([chunk, pdf_name])
        
        if all_chunks:
            self.vector_store.insert_data(
                ["content", "source"],
                all_chunks,
                [0]
            )
        
        print(f"✅ Đã index {len(all_chunks)} chunks từ {len(extracted_data)} PDF")
    
    def answer_question(self, question, options):
        """
        Trả lời câu hỏi trắc nghiệm
        
        Args:
            question: Câu hỏi
            options: Dict {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
        
        Returns:
            tuple: (số_câu_đúng, list_đáp_án_đúng)
        """
        # Tìm context liên quan
        search_results = self.vector_store.search(question, top_k=5, threshold=0.3)
        
        if not search_results:
            # Nếu không tìm thấy context, dùng LLM trực tiếp
            context = "Không tìm thấy thông tin liên quan."
        else:
            context = "\n\n".join([
                f"[Nguồn: {point.payload.get('source', 'Unknown')}]\n{point.payload.get('content', '')}"
                for point in search_results
            ])
        
        # Tạo prompt cho LLM
        prompt = self._create_qa_prompt(question, options, context)
        
        # Gọi LLM
        response = ask_llm(prompt)
        
        # Parse kết quả
        correct_count, correct_answers = self._parse_llm_response(response)
        
        return correct_count, correct_answers
    
    def _create_qa_prompt(self, question, options, context):
        """Tạo prompt cho LLM"""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi trắc nghiệm.

THÔNG TIN TÀI LIỆU:
{context}

CÂU HỎI:
{question}

CÁC LỰA CHỌN:
{options_text}

YÊU CẦU:
- Câu hỏi có thể có NHIỀU đáp án đúng (từ 1 đến 4 đáp án)
- Phân tích kỹ từng lựa chọn dựa trên thông tin tài liệu
- Trả lời theo định dạng:
  Số câu đúng: <số>
  Đáp án đúng: <A,B,C,D>

VÍ DỤ:
Số câu đúng: 2
Đáp án đúng: A,C

PHÂN TÍCH VÀ TRẢ LỜI:"""
        
        return prompt
    
    def _parse_llm_response(self, response):
        """
        Parse response từ LLM
        
        Returns:
            tuple: (correct_count, correct_answers_list)
        """
        import re
        
        # Tìm số câu đúng
        count_match = re.search(r'Số câu đúng:\s*(\d+)', response)
        correct_count = int(count_match.group(1)) if count_match else 1
        
        # Tìm đáp án đúng
        answers_match = re.search(r'Đáp án đúng:\s*([A-D,\s]+)', response)
        if answers_match:
            answers_str = answers_match.group(1).strip()
            correct_answers = [ans.strip() for ans in answers_str.split(',') if ans.strip()]
        else:
            # Fallback: tìm các chữ cái A-D trong response
            correct_answers = list(set(re.findall(r'\b([A-D])\b', response)))
            correct_answers.sort()
        
        # Đảm bảo số lượng đáp án khớp
        if len(correct_answers) != correct_count:
            correct_count = len(correct_answers)
        
        return correct_count, correct_answers
    
    def process_questions_csv(self, csv_path):
        """
        Xử lý tất cả câu hỏi từ CSV
        
        Args:
            csv_path: Đường dẫn file question.csv
        
        Returns:
            list: [(correct_count, correct_answers), ...]
        """
        df = pd.read_csv(csv_path)
        results = []
        
        print(f"\n🤔 Bắt đầu trả lời {len(df)} câu hỏi...")
        
        for idx, row in df.iterrows():
            question = row.iloc[0]  # Cột đầu tiên là câu hỏi
            options = {
                'A': row.iloc[1],
                'B': row.iloc[2],
                'C': row.iloc[3],
                'D': row.iloc[4]
            }
            
            print(f"\nCâu {idx + 1}/{len(df)}: {question[:50]}...")
            
            correct_count, correct_answers = self.answer_question(question, options)
            results.append((correct_count, correct_answers))
            
            print(f"  ➜ Số câu đúng: {correct_count}, Đáp án: {','.join(correct_answers)}")
        
        return results


def run_qa_pipeline(pdf_dir, question_csv, output_dir):
    """
    Chạy toàn bộ pipeline: Extract + Index + QA
    
    Args:
        pdf_dir: Thư mục chứa PDF
        question_csv: File CSV câu hỏi
        output_dir: Thư mục output
    """
    from extract_pdf import extract_all_pdfs
    
    # Bước 1: Trích xuất PDF
    print("=" * 60)
    print("BƯỚC 1: TRÍCH XUẤT PDF")
    print("=" * 60)
    extracted_data = extract_all_pdfs(pdf_dir, output_dir)
    
    # Bước 2: Index dữ liệu
    print("\n" + "=" * 60)
    print("BƯỚC 2: INDEX DỮ LIỆU")
    print("=" * 60)
    qa_system = QASystem(collection_name="technical_docs_qa")
    qa_system.index_extracted_data(extracted_data)
    
    # Bước 3: Trả lời câu hỏi
    print("\n" + "=" * 60)
    print("BƯỚC 3: TRẢ LỜI CÂU HỎI")
    print("=" * 60)
    results = qa_system.process_questions_csv(question_csv)
    
    return extracted_data, results


if __name__ == "__main__":
    # Test
    pdf_dir = "main/data/processed/public_test/pdfs"
    question_csv = "main/data/processed/public_test/question.csv"
    output_dir = "main/output/public_test_output"
    
    extracted_data, results = run_qa_pipeline(pdf_dir, question_csv, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH!")
    print("=" * 60)