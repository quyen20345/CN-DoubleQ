# src/pipeline/tasks.py
"""
Module này điều phối các tác vụ chính của pipeline: extract và qa.
"""
import traceback
import json
from pathlib import Path

from src.data_processing.pdf_parser import PDFMarkdownConverter
from src.embedding.model import EmbeddingModel
from src.vectordb.store import VectorStore
from src.vectordb.indexer import index_documents
from src.rag_system.qa_handler import QAHandler
from src.rag_system.retriever import HybridRetriever
from .output_generator import OutputGenerator

def run_extract_task(paths: dict) -> bool:
    """
    Chạy tác vụ trích xuất: đọc PDF, chunk, embed, và index.
    Lưu lại corpus để tác vụ QA có thể sử dụng cho BM25.
    """
    print("\n" + "="*25 + " BẮT ĐẦU TÁC VỤ EXTRACT " + "="*25)
    input_dir = Path(paths["pdf_dir"])
    output_dir = Path(paths["output_dir"])
    corpus_path = output_dir / "corpus.json"
    
    converter = PDFMarkdownConverter()
    extracted_data = {}

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào trong: {input_dir}")
        return False

    for pdf in pdf_files:
        pdf_output_sub_dir = output_dir / pdf.stem
        try:
            md_content, image_count = converter.convert(pdf, pdf_output_sub_dir)
            extracted_data[pdf.stem] = md_content
            print(f"✅ Trích xuất thành công: {pdf.name} ({image_count} ảnh)")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {pdf.name}: {e}")
            traceback.print_exc()

    if not extracted_data:
        print("❌ Không có file PDF nào được xử lý thành công.")
        return False

    embedding_model = EmbeddingModel()
    collection_name = f"collection_{input_dir.name}"
    vector_db = VectorStore(collection_name, embedding_model)
    
    # Index dữ liệu và lấy lại corpus
    corpus_for_bm25 = index_documents(extracted_data, vector_db)

    # Lưu corpus cho tác vụ QA
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_for_bm25, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu corpus cho BM25 vào: {corpus_path}")

    print("\n" + "="*24 + " HOÀN THÀNH TÁC VỤ EXTRACT " + "="*24)
    return True

def run_qa_task(paths: dict):
    """
    Chạy tác vụ trả lời câu hỏi: tải corpus, khởi tạo retriever, và xử lý câu hỏi.
    """
    print("\n" + "="*28 + " BẮT ĐẦU TÁC VỤ QA " + "="*28)
    output_dir = Path(paths["output_dir"])
    corpus_path = output_dir / "corpus.json"

    # Tải corpus đã được xử lý từ tác vụ extract
    if not corpus_path.exists():
        print(f"❌ Không tìm thấy file corpus.json. Vui lòng chạy tác vụ 'extract' trước.")
        return
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    if not corpus_data:
        print("❌ Dữ liệu corpus trống. Không thể tiếp tục.")
        return

    # Khởi tạo các thành phần cần thiết
    embedding_model = EmbeddingModel()
    collection_name = f"collection_{Path(paths['pdf_dir']).name}"
    vector_db = VectorStore(collection_name, embedding_model)
    
    # Khởi tạo Hybrid Retriever
    retriever = HybridRetriever(vector_db, corpus_data)
    
    # Khởi tạo QA Handler với retriever
    qa_handler = QAHandler(retriever)
    
    # Xử lý các câu hỏi
    qa_results = qa_handler.process_questions_csv(paths["question_csv"])
    if qa_results is None:
        return

    # Tạo file output
    generator = OutputGenerator(output_dir)
    # Lấy lại dữ liệu đã trích xuất từ các file main.md để tạo output
    extracted_md_data = {}
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            md_file = subdir / "main.md"
            if md_file.exists():
                extracted_md_data[subdir.name] = md_file.read_text(encoding="utf-8")

    generator.generate_final_output(extracted_md_data, qa_results, paths["zip_name"])
    
    print("\n" + "="*27 + " HOÀN THÀNH TÁC VỤ QA " + "="*27)

