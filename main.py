# main.py
"""
Đây là entry point chính của toàn bộ pipeline.
Nó chịu trách nhiệm phân tích các đối số dòng lệnh (command-line arguments)
và gọi các tác vụ tương ứng trong module pipeline.
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục `src` vào Python path để có thể import các module
# một cách nhất quán từ thư mục gốc của dự án.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.paths import setup_project_paths
from src.pipeline.tasks import run_extract_task, run_qa_task

def main():
    """
    Hàm chính điều phối toàn bộ pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline cho Nhiệm vụ 2: Khai phá tri thức từ văn bản kỹ thuật - Zalo AI Challenge",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", 
        choices=["public", "private", "training"], 
        default="public",
        help="Chọn chế độ chạy:\n"
             " - public: Sử dụng dữ liệu public test\n"
             " - private: Sử dụng dữ liệu private test\n"
             " - training: Sử dụng dữ liệu training"
    )
    parser.add_argument(
        "--task", 
        choices=["extract", "qa", "full"], 
        default="full",
        help="Chọn tác vụ cần thực hiện:\n"
             " - extract: Chỉ trích xuất, chunk, và index dữ liệu từ PDF.\n"
             " - qa: Chỉ chạy phần trả lời câu hỏi (yêu cầu đã chạy extract trước).\n"
             " - full: Chạy toàn bộ pipeline từ đầu đến cuối (mặc định)."
    )
    args = parser.parse_args()

    print(f"\n{'*'*80}\n{' BẮT ĐẦU PIPELINE '.center(80,'*')}\n{'*'*80}")
    print(f"Chế độ: {args.mode.upper()} | Tác vụ: {args.task.upper()}")

    try:
        # Thiết lập đường dẫn
        paths = setup_project_paths(args.mode)

        # Thực thi tác vụ
        if args.task == "extract":
            run_extract_task(paths)
        elif args.task == "qa":
            run_qa_task(paths)
        elif args.task == "full":
            # Chạy extract, nếu thành công thì chạy tiếp qa
            extract_success = run_extract_task(paths)
            if extract_success:
                run_qa_task(paths)
            else:
                print("\n❌ Tác vụ 'extract' thất bại. Tác vụ 'qa' sẽ không được thực hiện.")

    except FileNotFoundError as e:
        print(f"\n❌ Lỗi nghiêm trọng: {e}")
        print("Vui lòng đảm bảo bạn đã chạy 'bash scripts/prepare_data.sh' và dữ liệu tồn tại đúng chỗ.")
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi không mong muốn: {e}")

    print(f"\n{'*'*80}\n{' PIPELINE KẾT THÚC '.center(80,'*')}\n{'*'*80}\n")


if __name__ == "__main__":
    main()
