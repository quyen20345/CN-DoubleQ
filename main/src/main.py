# -*- coding: utf-8 -*-
"""
Tệp mã nguồn tổng hợp cho Nhiệm vụ 2: Khai phá tri thức từ văn bản kỹ thuật.

Tệp này chứa toàn bộ pipeline, bao gồm:
1. Trích xuất dữ liệu từ file PDF sang định dạng Markdown.
2. Lập chỉ mục (index) nội dung đã trích xuất vào Vector Database (Qdrant).
3. Hệ thống Hỏi-Đáp (QA) để trả lời các câu hỏi trắc nghiệm.
4. Tạo tệp output cuối cùng (`answer.md` và file zip) theo đúng định dạng yêu cầu.

Để chạy chương trình, sử dụng command line với các tùy chọn phù hợp.
Ví dụ:
- Chạy toàn bộ pipeline cho public test:
  python3 main.py --mode public --task full

- Chỉ chạy phần trích xuất và index cho public test:
  python3 main.py --mode public --task extract

- Chỉ chạy phần trả lời câu hỏi cho public test (sau khi đã extract):
  python3 main.py --mode public --task qa
"""

# ==============================================================================
# 0. IMPORTS & SETUP
# ==============================================================================
import os
import re
import sys
import shutil
import argparse
import uuid
import zipfile
from pathlib import Path
import pandas as pd

# Cài đặt các thư viện cần thiết
# pip install "python-dotenv==1.0.1" "pandas==2.2.2" "sentence-transformers==3.0.1" "qdrant-client==1.9.2" "langchain==0.2.6" "langchain-ollama==0.1.0" "jinja2==3.1.4" "docling==0.0.15" "tiktoken==0.7.0"

# Tải .env file nếu có
try:
    from dotenv import load_dotenv
    if Path('.env').exists():
        print("Đang tải biến môi trường từ file .env...")
        load_dotenv()
except ImportError:
    print("Cảnh báo: Thư viện python-dotenv chưa được cài. Sử dụng biến môi trường hệ thống.")

# ==============================================================================
# 1. EMBEDDING MODEL (từ main/src/embedding/model.py)
# ==============================================================================
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Lỗi: sentence-transformers chưa được cài. Vui lòng chạy: pip install sentence-transformers")
    sys.exit(1)

class DenseEmbedding:
    """Quản lý mô hình embedding để chuyển văn bản thành vector."""
    def __init__(self, model_name=os.getenv("DENSE_MODEL", "vinai/phobert-base-v2")):
        print(f"Đang khởi tạo mô hình embedding: {model_name}")
        # Tạo thư mục cache trong project để lưu model
        cache_dir = Path(__file__).parent / "embedding_models"
        cache_dir.mkdir(exist_ok=True)
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        print("✅ Khởi tạo mô hình embedding thành công.")

    def encode(self, texts):
        if isinstance(texts, str):
            return self.model.encode(texts)
        elif isinstance(texts, list):
            return [e.tolist() for e in self.model.encode(texts)]
        else:
            raise ValueError("Đầu vào phải là chuỗi hoặc danh sách các chuỗi.")
        
    def get_dimension(self):
        # Trả về số chiều của vector embedding
        return self.model.get_sentence_embedding_dimension()

# ==============================================================================
# 2. VECTOR DATABASE (từ main/src/vectordb/qdrant.py)
# ==============================================================================
try:
    from qdrant_client import QdrantClient, models
except ImportError:
    print("Lỗi: qdrant-client chưa được cài. Vui lòng chạy: pip install qdrant-client")
    sys.exit(1)

class VectorStore:
    """Quản lý việc lưu trữ và truy vấn vector tại Qdrant."""
    def __init__(self, collection_name, dense_model):
        self.collection_name = collection_name
        self.dense_embedding_model = dense_model
        
        try:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", 6333))
            print(f"Đang kết nối tới Qdrant tại {host}:{port}...")
            self.client = QdrantClient(host=host, port=port, timeout=int(os.getenv("QDRANT_TIMEOUT", 60)))
            self.client.get_collections() # Kiểm tra kết nối
            print("✅ Kết nối Qdrant thành công.")
        except Exception as e:
            print(f"❌ Không thể kết nối tới Qdrant: {e}")
            print("Hãy chắc chắn rằng bạn đã khởi chạy Qdrant (ví dụ: bằng docker-compose).")
            sys.exit(1)

        if not self.client.collection_exists(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        print(f"Đang tạo collection mới: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dense_embedding_model.get_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

    def recreate_collection(self):
        print(f"Đang xóa và tạo lại collection: {self.collection_name}")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._create_collection()
        print("✅ Tạo lại collection thành công.")

    def insert_data(self, payload_keys, payload_values):
        if not payload_values:
            return
        
        contents = [item[0] for item in payload_values]
        dense_embeddings = self.dense_embedding_model.encode(contents)

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=dense_embeddings[i],
                payload=dict(zip(payload_keys, payload_values[i]))
            )
            for i in range(len(dense_embeddings))
        ]
        
        # Upsert theo batch
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            print(f"  > Đã upsert batch {i // BATCH_SIZE + 1}/{len(points) // BATCH_SIZE + 1}")

    def search(self, query, top_k=5, threshold=0.3):
        query_vector = self.dense_embedding_model.encode(query)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=threshold
        )
        return hits

# ==============================================================================
# 3. LLM INTEGRATION (từ main/src/llm/*.py)
# ==============================================================================
try:
    from langchain_ollama import OllamaLLM
    from jinja2 import Template
except ImportError:
    print("Lỗi: langchain-ollama hoặc jinja2 chưa được cài. Vui lòng cài đặt.")
    sys.exit(1)

class LLMManager:
    """Quản lý việc tương tác với Large Language Model."""
    def __init__(self):
        llm_type = os.getenv("LLM_TYPE", "ollama")
        if llm_type != "ollama":
            raise ValueError("Hiện chỉ hỗ trợ LLM_TYPE=ollama")
        
        model_name = os.getenv("CHAT_MODEL", "llama3:instruct")
        print(f"Đang khởi tạo LLM: {model_name}")
        self.llm = OllamaLLM(model=model_name, temperature=0.0)
        print("✅ Khởi tạo LLM thành công.")

    def ask(self, prompt):
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"❌ Lỗi khi gọi LLM: {e}")
            print("Hãy chắc chắn rằng Ollama đang chạy và model đã được pull (vd: ollama run llama3:instruct).")
            # Fallback response
            return "Số câu đúng: 1\nĐáp án đúng: A"

# ==============================================================================
# 4. PDF EXTRACTION (từ extract_pdf.py)
# ==============================================================================
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
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    
    def extract_pdf(self, pdf_path, output_dir):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"Đang xử lý PDF: {pdf_path}")
        result = self.converter.convert(str(pdf_path))
        doc = result.document
        
        # Lưu hình ảnh và công thức (nếu có)
        # Theo yêu cầu, chúng ta chỉ cần placeholder
        if hasattr(doc, 'pictures'):
            for i, picture in enumerate(doc.pictures):
                try:
                    if hasattr(picture, 'image') and picture.image:
                        img_path = images_dir / f"image_{i+1}.png"
                        picture.image.pil_image.save(img_path)
                except Exception as e:
                    print(f"  > Cảnh báo: Không thể lưu ảnh {i+1}. Lỗi: {e}")

        # Export ra markdown và thay thế placeholders
        md_content = doc.export_to_markdown()
        md_content = re.sub(r'!\[.*?\]\(data:image/.*?\)', r'|<image_placeholder>|', md_content)
        md_content = re.sub(r'<img src="data:image/.*?">', r'|<image_placeholder>|', md_content)
        
        # Đánh số lại placeholders
        image_counter = 1
        formula_counter = 1
        
        def replace_image(match):
            nonlocal image_counter
            res = f"|<image_{image_counter}>|"
            image_counter += 1
            return res

        def replace_formula(match):
            nonlocal formula_counter
            res = f"|<formula_{formula_counter}>|"
            formula_counter += 1
            return res
            
        md_content = re.sub(r'\|<image_placeholder>\|', replace_image, md_content)
        md_content = re.sub(r'\$\$[\s\S]*?\$\$', replace_formula, md_content) # Block formula
        md_content = re.sub(r'\$[^\$]*?\$', replace_formula, md_content) # Inline formula

        main_md_path = output_dir_path / "main.md"
        with open(main_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ Đã trích xuất xong: {main_md_path}")
        return md_content

    def extract_all_pdfs(self, input_dir, output_base_dir):
        extracted_data = {}
        pdf_files = list(Path(input_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"Cảnh báo: Không tìm thấy file PDF nào trong '{input_dir}'")
            return {}

        for pdf_path in pdf_files:
            pdf_name = pdf_path.stem
            output_dir = Path(output_base_dir) / pdf_name
            markdown_content = self.extract_pdf(pdf_path, output_dir)
            extracted_data[pdf_name] = markdown_content
        return extracted_data

# ==============================================================================
# 5. QA SYSTEM (từ qa_system.py và _utils.py)
# ==============================================================================
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Lỗi: langchain chưa được cài. Vui lòng chạy: pip install langchain tiktoken")
    sys.exit(1)

def chunking(text):
    """Phân nhỏ văn bản thành các đoạn có kích thước phù hợp."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk.strip()) >= 50]

class QASystem:
    """Hệ thống trả lời câu hỏi trắc nghiệm."""
    def __init__(self, vector_store, llm_manager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager
    
    def index_data(self, extracted_data):
        print("🔄 Đang index dữ liệu vào vector database...")
        self.vector_store.recreate_collection()
        
        all_chunks = []
        for pdf_name, content in extracted_data.items():
            chunks = chunking(content)
            for chunk in chunks:
                all_chunks.append([chunk, pdf_name])
        
        if all_chunks:
            self.vector_store.insert_data(["content", "source"], all_chunks)
        
        print(f"✅ Đã index {len(all_chunks)} chunks từ {len(extracted_data)} PDF.")
    
    def answer_question(self, question, options):
        # Tìm kiếm context liên quan
        search_results = self.vector_store.search(question, top_k=5, threshold=0.3)
        
        context = "\n\n---\n\n".join([
            f"Nguồn: {point.payload.get('source', 'Không rõ')}\n\n{point.payload.get('content', '')}"
            for point in search_results
        ]) if search_results else "Không có thông tin nào được tìm thấy trong tài liệu."
        
        prompt = self._create_qa_prompt(question, options, context)
        response = self.llm_manager.ask(prompt)
        return self._parse_llm_response(response)
    
    def _create_qa_prompt(self, question, options, context):
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

Ví dụ:
{{
  "correct_count": 2,
  "correct_answers": ["A", "C"]
}}

### TRẢ LỜI (CHỈ JSON):
"""

    def _parse_llm_response(self, response):
        try:
            import json
            # Tìm và trích xuất chuỗi JSON từ response
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                count = data.get("correct_count", 0)
                answers = data.get("correct_answers", [])
                
                # Xác thực lại dữ liệu
                answers = [ans for ans in answers if ans in ['A', 'B', 'C', 'D']]
                if count != len(answers):
                    print(f"  > Cảnh báo: LLM trả về số lượng không khớp. count={count}, answers={answers}. Tự động sửa lại.")
                    count = len(answers)
                
                return count, answers
            else:
                raise ValueError("Không tìm thấy JSON trong response.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  > Cảnh báo: Không thể parse JSON từ LLM. Lỗi: {e}. Response: '{response[:100]}...'")
            # Fallback: Dùng regex để tìm câu trả lời
            answers = sorted(list(set(re.findall(r'\b([A-D])\b', response.upper()))))
            return len(answers), answers
    
    def process_questions_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy file question.csv tại '{csv_path}'")
            return None
            
        results = []
        print(f"\n🤔 Bắt đầu trả lời {len(df)} câu hỏi...")
        
        for idx, row in df.iterrows():
            question = row.iloc[0]
            options = { 'A': row.iloc[1], 'B': row.iloc[2], 'C': row.iloc[3], 'D': row.iloc[4] }
            
            print(f"\nCâu {idx + 1}/{len(df)}: {question[:80]}...")
            
            count, answers = self.answer_question(question, options)
            results.append((count, answers))
            
            print(f"  ➜ Kết quả: {count} câu đúng - Đáp án: {', '.join(answers) if answers else 'Không có'}")
        
        return results

# ==============================================================================
# 6. ANSWER GENERATOR (từ answer_generator.py)
# ==============================================================================
class AnswerGenerator:
    """Tạo file output cuối cùng theo định dạng của cuộc thi."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_answer_md(self, extracted_data, qa_results):
        content = []
        
        # Phần 1: TASK EXTRACT
        content.append("### TASK EXTRACT")
        for pdf_name, md_content in sorted(extracted_data.items()):
            content.append(f"\n# {pdf_name}\n")
            content.append(md_content)
        
        # Phần 2: TASK QA
        content.append("\n### TASK QA\n")
        if qa_results:
            for count, answers in qa_results:
                content.append(f"{count}")
                content.append(f"{','.join(sorted(answers))}")
        
        answer_md_path = self.output_dir / "answer.md"
        with open(answer_md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"✅ Đã tạo file answer.md tại: {answer_md_path}")

    def create_zip(self, zip_name):
        zip_path = self.output_dir.parent / zip_name
        print(f"📦 Đang tạo file zip: {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, self.output_dir)
                    zipf.write(file_path, archive_name)

        print(f"✅ Đã tạo file zip thành công: {zip_path}")

# ==============================================================================
# 7. MAIN PIPELINE LOGIC
# ==============================================================================
def setup_paths(mode):
    """Thiết lập đường dẫn input và output dựa trên mode."""
    base_input_dir = Path(f"data/{mode}_test_input")
    output_dir = Path(f"output/{mode}_test_output")
    
    # Tạo các thư mục nếu chưa tồn tại
    base_input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "pdf_dir": base_input_dir,
        "question_csv": base_input_dir / "question.csv",
        "output_dir": output_dir,
        "zip_name": f"{mode}_test_output.zip"
    }
    
    print("\n--- Cấu hình đường dẫn ---")
    for key, value in paths.items():
        print(f"{key:<15}: {value}")
    print("---------------------------\n")
    
    return paths

def run_task_extract(paths):
    """Chạy tác vụ trích xuất và index."""
    print("\n" + "="*25 + " BẮT ĐẦU TÁC VỤ EXTRACT " + "="*25)
    
    # 1. Khởi tạo các thành phần
    extractor = PDFExtractor()
    embedding_model = DenseEmbedding()
    vector_db = VectorStore(f"collection_{Path(paths['pdf_dir']).name}", embedding_model)
    qa_system = QASystem(vector_db, None) # Không cần LLM cho task này

    # 2. Trích xuất PDF
    extracted_data = extractor.extract_all_pdfs(paths["pdf_dir"], paths["output_dir"])
    if not extracted_data:
        print("❌ Kết thúc tác vụ extract vì không có dữ liệu PDF.")
        return False

    # 3. Index dữ liệu
    qa_system.index_data(extracted_data)
    
    print("\n" + "="*25 + " HOÀN THÀNH TÁC VỤ EXTRACT " + "="*24)
    return True

def run_task_qa(paths):
    """Chạy tác vụ trả lời câu hỏi và tạo output."""
    print("\n" + "="*28 + " BẮT ĐẦU TÁC VỤ QA " + "="*28)
    
    # 1. Khởi tạo các thành phần
    embedding_model = DenseEmbedding()
    vector_db = VectorStore(f"collection_{Path(paths['pdf_dir']).name}", embedding_model)
    llm = LLMManager()
    qa_system = QASystem(vector_db, llm)

    # 2. Đọc lại dữ liệu đã trích xuất từ file main.md
    extracted_data = {}
    output_dir_path = Path(paths["output_dir"])
    for subdir in output_dir_path.iterdir():
        if subdir.is_dir():
            main_md = subdir / "main.md"
            if main_md.exists():
                pdf_name = subdir.name
                extracted_data[pdf_name] = main_md.read_text(encoding='utf-8')

    if not extracted_data:
        print("❌ Lỗi: Không tìm thấy dữ liệu đã trích xuất trong thư mục output.")
        print("Vui lòng chạy tác vụ 'extract' trước: python3 main.py --task extract")
        return

    # 3. Trả lời câu hỏi
    qa_results = qa_system.process_questions_csv(paths["question_csv"])
    if qa_results is None:
        print("❌ Kết thúc tác vụ QA vì không thể xử lý file câu hỏi.")
        return

    # 4. Tạo file output
    generator = AnswerGenerator(paths["output_dir"])
    generator.generate_answer_md(extracted_data, qa_results)
    
    # 5. Copy file main.py và tạo zip
    print(f"Đang copy file mã nguồn '{Path(__file__).name}' vào thư mục output...")
    shutil.copy(Path(__file__), output_dir_path / "main.py")
    generator.create_zip(paths["zip_name"])
    
    print("\n" + "="*27 + " HOÀN THÀNH TÁC VỤ QA " + "="*28)


def main():
    """Hàm main để chạy pipeline từ command line."""
    parser = argparse.ArgumentParser(description="Pipeline cho Nhiệm vụ 2 - Zalo AI Challenge")
    
    parser.add_argument(
        "--mode", type=str, choices=["public", "private", "training"], default="public",
        help="Chế độ chạy: public, private hoặc training test."
    )
    parser.add_argument(
        "--task", type=str, choices=["extract", "qa", "full"], default="full",
        help="Tác vụ cần thực hiện: extract (trích xuất & index), qa (trả lời câu hỏi), full (toàn bộ pipeline)."
    )
    
    args = parser.parse_args()
    
    print("\n" + "*"*80)
    print(f" BẮT ĐẦU PIPELINE - CHẾ ĐỘ: {args.mode.upper()} - TÁC VỤ: {args.task.upper()} ".center(80, '*'))
    print("*"*80)

    # Thiết lập đường dẫn
    paths = setup_paths(args.mode)

    if args.task == "extract":
        run_task_extract(paths)
    elif args.task == "qa":
        run_task_qa(paths)
    elif args.task == "full":
        if run_task_extract(paths):
            run_task_qa(paths)
    
    print("\n" + "*"*80)
    print(f" PIPELINE KẾT THÚC ".center(80, '*'))
    print("*"*80 + "\n")


if __name__ == "__main__":
    # Đặt thư mục làm việc là thư mục chứa file script
    os.chdir(Path(__file__).parent)
    main()
