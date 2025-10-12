# -*- coding: utf-8 -*-
"""
Tệp mã nguồn chính cho Nhiệm vụ 2: Khai phá tri thức từ văn bản kỹ thuật.

Đây là điểm khởi đầu (entry point) của chương trình, chịu trách nhiệm:
1. Phân tích các tham số dòng lệnh (--mode, --task).
2. Điều phối (orchestrate) các tác vụ bằng cách gọi các module xử lý tương ứng.
"""
import argparse
import sys
import shutil
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để import các module khác cấp
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from main.extract_pdf import PDFToMarkdownConverter as PDFExtractor
from main.src.embedding.model import DenseEmbedding
from main.src.vectordb.qdrant import VectorStore
from main.src.utils.indexer import index_extracted_data
# from main.src.llm.chat import QAHandler
from main.src.answer_generator import QAHandler, AnswerGenerator

def setup_paths(mode: str) -> dict:
    """Thiết lập và xác thực các đường dẫn input và output."""
    # Xử lý cấu trúc thư mục khác nhau
    if mode == "public":
        base_input_dir = project_root / f"main/data/{mode}_test_input/{mode}-test-input"
    elif mode == "private":
        base_input_dir = project_root / f"main/data/{mode}_test_input/{mode}_test_input"
    else:  # training
        base_input_dir = project_root / f"main/data/{mode}_test_input/{mode}_input"

    output_dir = project_root / f"output/{mode}_test_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Tìm file question.csv trong tất cả thư mục con
        question_csv_path = next(base_input_dir.rglob("question.csv"))
    except StopIteration:
        print(f"❌ Lỗi nghiêm trọng: Không tìm thấy 'question.csv' trong thư mục '{base_input_dir}'.")
        sys.exit(1)

    paths = {
        "pdf_dir": base_input_dir,
        "question_csv": question_csv_path,
        "output_dir": output_dir,
        "zip_name": f"{mode}_test_output.zip"
    }
    
    print("\n--- Cấu hình đường dẫn ---")
    for key, value in paths.items():
        try:
            print(f"{key:<15}: {Path(value).relative_to(project_root)}")
        except (TypeError, ValueError):
            print(f"{key:<15}: {value}")
    print("---------------------------\n")

    return paths

def run_task_extract(paths: dict) -> bool:
    """Chạy tác vụ trích xuất PDF bằng module extract_pdf.py và index dữ liệu."""
    print("\n" + "="*25 + " BẮT ĐẦU TÁC VỤ EXTRACT " + "="*25)
    
    from main.extract_pdf import PDFToMarkdownConverter  # dùng converter mới

    converter = PDFToMarkdownConverter()
    input_dir = Path(paths["pdf_dir"])
    
    # Output sẽ nằm trong: output/public_test_output/
    base_output_dir = Path(paths["output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)

    extracted_data = {}

    for pdf_file in input_dir.glob("*.pdf"):
        # Mỗi file PDF có thư mục riêng: output/public_test_output/<PDF_NAME>/images/
        pdf_output_dir = base_output_dir / pdf_file.stem / "images"
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            md_path = converter.convert_pdf_to_markdown(str(pdf_file), str(pdf_output_dir))
            # Lưu text markdown đã trích xuất để index
            if md_path and Path(md_path).exists():
                extracted_data[pdf_file.stem] = Path(md_path).read_text(encoding="utf-8")
            print(f"✅ Trích xuất thành công: {pdf_file.name}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()

    if not extracted_data:
        print("❌ Không có file PDF nào được xử lý thành công.")
        return False

    # Index dữ liệu vào vector DB (giữ nguyên pipeline cũ)
    embedding_model = DenseEmbedding()
    collection_name = f"collection_{paths['pdf_dir'].name}"
    vector_db = VectorStore(collection_name, embedding_model)
    index_extracted_data(extracted_data, vector_db)

    print("\n" + "="*24 + " HOÀN THÀNH TÁC VỤ EXTRACT " + "="*24)
    return True


def run_task_qa(paths: dict):
    """Chạy tác vụ trả lời câu hỏi và tạo file nộp bài."""
    print("\n" + "="*28 + " BẮT ĐẦU TÁC VỤ QA " + "="*28)
    
    embedding_model = DenseEmbedding()
    collection_name = f"collection_{paths['pdf_dir'].name}"
    vector_db = VectorStore(collection_name, embedding_model)
    qa_handler = QAHandler(vector_db)

    # Đọc lại dữ liệu đã trích xuất từ các file main.md
    extracted_data = {}
    output_dir_path = Path(paths["output_dir"])
    
    print(f"🔍 Tìm kiếm dữ liệu trong: {output_dir_path}")
    
    for subdir in output_dir_path.iterdir():
        if subdir.is_dir():
            # Tìm main.md trong thư mục gốc trước
            main_md = subdir / "main.md"
            if main_md.exists():
                print(f"✅ Tìm thấy: {subdir.name}/main.md")
                extracted_data[subdir.name] = main_md.read_text(encoding='utf-8')
            else:
                # Tìm main.md trong thư mục images (fallback)
                images_md = subdir / "images" / "main.md"
                if images_md.exists():
                    print(f"✅ Tìm thấy: {subdir.name}/images/main.md")
                    extracted_data[subdir.name] = images_md.read_text(encoding='utf-8')
                else:
                    print(f"⚠️ Không tìm thấy main.md trong {subdir.name}")

    if not extracted_data:
        print(f"❌ Lỗi: Không tìm thấy dữ liệu đã trích xuất trong '{output_dir_path}'.")
        print("Vui lòng chạy tác vụ 'extract' trước.")
        return

    print(f"📊 Đã tìm thấy {len(extracted_data)} file dữ liệu")

    # Trả lời câu hỏi
    qa_results = qa_handler.process_questions_csv(paths["question_csv"])
    if qa_results is None:
        return

    # Tạo file kết quả và đóng gói
    generator = AnswerGenerator(output_dir_path)
    generator.generate_answer_md(extracted_data, qa_results)
    
    # Copy file main.py này vào thư mục output trước khi nén
    this_script_path = project_root / "main" / "src" / "main.py"
    shutil.copy(this_script_path, output_dir_path / "main.py")
    
    generator.create_zip(paths["zip_name"])
    
    print("\n" + "="*27 + " HOÀN THÀNH TÁC VỤ QA " + "="*27)

def main():
    """Hàm chính, phân tích tham số và điều phối pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline cho Nhiệm vụ 2 - Zalo AI Challenge")
    parser.add_argument(
        "--mode", type=str, choices=["public", "private", "training"], default="public",
        help="Chế độ chạy: public, private hoặc training."
    )
    parser.add_argument(
        "--task", type=str, choices=["extract", "qa", "full"], default="full",
        help="Tác vụ cần thực hiện: extract, qa, hoặc full."
    )
    args = parser.parse_args()
    
    print("\n" + "*"*80)
    print(f" BẮT ĐẦU PIPELINE - CHẾ ĐỘ: {args.mode.upper()} - TÁC VỤ: {args.task.upper()} ".center(80, '*'))
    print("*"*80)

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
    main()
