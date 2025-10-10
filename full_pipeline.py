"""
Pipeline đầy đủ cho nhiệm vụ 2:
1. Trích xuất PDF → Markdown
2. Index dữ liệu vào vector database
3. Trả lời câu hỏi trắc nghiệm
4. Tạo file answer.md và zip
"""

import sys
import argparse
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

from extract_pdf import extract_all_pdfs
from qa_system import QASystem
from answer_generator import create_complete_output


def run_full_pipeline(
    pdf_dir: str,
    question_csv: str,
    output_dir: str,
    create_zip: bool = True,
    zip_name: str = None
):
    """
    Chạy toàn bộ pipeline
    
    Args:
        pdf_dir: Thư mục chứa file PDF
        question_csv: File CSV chứa câu hỏi
        output_dir: Thư mục output
        create_zip: Có tạo file zip không
        zip_name: Tên file zip (mặc định: <output_dir_name>.zip)
    """
    
    print("\n" + "=" * 70)
    print("  PIPELINE TRÍCH XUẤT VÀ TRẢ LỜI CÂU HỎI - NHIỆM VỤ 2")
    print("=" * 70)
    print(f"📁 PDF Directory:     {pdf_dir}")
    print(f"📄 Question CSV:      {question_csv}")
    print(f"📂 Output Directory:  {output_dir}")
    print("=" * 70 + "\n")
    
    # ========== BƯỚC 1: TRÍCH XUẤT PDF ==========
    print("🔵 BƯỚC 1: TRÍCH XUẤT PDF SANG MARKDOWN")
    print("-" * 70)
    
    extracted_data = extract_all_pdfs(pdf_dir, output_dir)
    
    if not extracted_data:
        print("❌ Không tìm thấy file PDF nào để trích xuất!")
        return
    
    print(f"✅ Đã trích xuất {len(extracted_data)} file PDF\n")
    
    # ========== BƯỚC 2: INDEX DỮ LIỆU ==========
    print("🔵 BƯỚC 2: INDEX DỮ LIỆU VÀO VECTOR DATABASE")
    print("-" * 70)
    
    qa_system = QASystem(collection_name="technical_docs_qa")
    qa_system.index_extracted_data(extracted_data)
    print()
    
    # ========== BƯỚC 3: TRẢ LỜI CÂU HỎI ==========
    print("🔵 BƯỚC 3: TRẢ LỜI CÂU HỎI TRẮC NGHIỆM")
    print("-" * 70)
    
    qa_results = qa_system.process_questions_csv(question_csv)
    print(f"\n✅ Đã trả lời {len(qa_results)} câu hỏi\n")
    
    # ========== BƯỚC 4: TẠO FILE ANSWER.MD ==========
    print("🔵 BƯỚC 4: TẠO FILE KẾT QUẢ")
    print("-" * 70)
    
    # Tìm file main.py để copy
    main_py_candidates = [
        "main.py",
        "main/src/main.py",
        str(Path(__file__).parent / "main.py")
    ]
    
    main_py_path = None
    for candidate in main_py_candidates:
        if Path(candidate).exists():
            main_py_path = candidate
            break
    
    generator = create_complete_output(
        extracted_data,
        qa_results,
        output_dir,
        main_py_path
    )
    
    # ========== BƯỚC 5: TẠO FILE ZIP (NẾU CẦN) ==========
    if create_zip:
        print("\n🔵 BƯỚC 5: TẠO FILE ZIP")
        print("-" * 70)
        
        if zip_name is None:
            zip_name = f"{Path(output_dir).name}.zip"
        
        generator.create_zip(zip_name)
    
    # ========== KẾT THÚC ==========
    print("\n" + "=" * 70)
    print("  ✅ HOÀN THÀNH TOÀN BỘ PIPELINE!")
    print("=" * 70)
    print(f"📂 Kết quả tại:       {output_dir}")
    print(f"📄 File answer.md:    {Path(output_dir) / 'answer.md'}")
    if create_zip:
        print(f"📦 File zip:          {Path(output_dir).parent / zip_name}")
    print("=" * 70 + "\n")


def main():
    """Main function với command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pipeline trích xuất PDF và trả lời câu hỏi trắc nghiệm"
    )
    
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="main/data/processed/public_test/pdfs",
        help="Thư mục chứa các file PDF cần trích xuất"
    )
    
    parser.add_argument(
        "--question_csv",
        type=str,
        default="main/data/processed/public_test/question.csv",
        help="File CSV chứa câu hỏi trắc nghiệm"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="main/output/public_test_output",
        help="Thư mục output"
    )
    
    parser.add_argument(
        "--create_zip",
        action="store_true",
        default=True,
        help="Tạo file zip từ output"
    )
    
    parser.add_argument(
        "--zip_name",
        type=str,
        default=None,
        help="Tên file zip (mặc định: <output_dir_name>.zip)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["public", "private"],
        default="public",
        help="Chế độ: public hoặc private test"
    )
    
    args = parser.parse_args()
    
    # Tự động điều chỉnh đường dẫn theo mode
    if args.mode == "private":
        args.pdf_dir = args.pdf_dir.replace("public_test", "private_test")
        args.question_csv = args.question_csv.replace("public_test", "private_test")
        args.output_dir = args.output_dir.replace("public_test_output", "private_test_output")
        if args.zip_name is None:
            args.zip_name = "private_test_output.zip"
    elif args.zip_name is None:
        args.zip_name = "public_test_output.zip"
    
    # Chạy pipeline
    run_full_pipeline(
        pdf_dir=args.pdf_dir,
        question_csv=args.question_csv,
        output_dir=args.output_dir,
        create_zip=args.create_zip,
        zip_name=args.zip_name
    )


if __name__ == "__main__":
    main()
    
    # Hoặc chạy trực tiếp với config mặc định:
    # run_full_pipeline(
    #     pdf_dir="main/data/processed/public_test/pdfs",
    #     question_csv="main/data/processed/public_test/question.csv",
    #     output_dir="main/output/public_test_output",
    #     create_zip=True,
    #     zip_name="public_test_output.zip"
    # )