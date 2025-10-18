# src/config/paths.py
"""
Module này chịu trách nhiệm thiết lập và quản lý tất cả các đường dẫn cần thiết cho dự án.
Việc tập trung quản lý đường dẫn ở một nơi giúp dễ dàng thay đổi và bảo trì.
"""

from pathlib import Path
import sys

# Thêm thư mục gốc của dự án vào sys.path để dễ dàng import
# Giả định rằng file này nằm trong src/config/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def setup_project_paths(mode: str) -> dict:
    """
    Thiết lập và xác thực các đường dẫn input và output dựa trên mode được chọn.
    Args:
        mode (str): Chế độ chạy ('public', 'private', 'training').
    Returns:
        dict: Một dictionary chứa các đường dẫn đã được cấu hình.
    """
    if mode == "public":
        base_input_dir = PROJECT_ROOT / f"data/{mode}_test_input/{mode}-test-input"
    elif mode == "private":
        base_input_dir = PROJECT_ROOT / f"data/{mode}_test_input/{mode}_test_input"
    else:
        # Giả định mode 'training' có cấu trúc tương tự
        base_input_dir = PROJECT_ROOT / f"data/{mode}_input/{mode}_input"

    output_dir = PROJECT_ROOT / f"output/{mode}_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    question_csv = next(base_input_dir.rglob("question.csv"), None)
    if not question_csv:
        raise FileNotFoundError(f"Không tìm thấy 'question.csv' trong {base_input_dir}")

    paths = {
        "project_root": PROJECT_ROOT,
        "pdf_dir": base_input_dir,
        "question_csv": question_csv,
        "output_dir": output_dir,
        "zip_name": f"{mode}_test_output.zip",
    }

    print("\n--- Cấu hình đường dẫn ---")
    for key, value in paths.items():
        try:
            # Hiển thị đường dẫn tương đối để dễ đọc
            print(f"{key:<15}: {Path(value).relative_to(PROJECT_ROOT)}")
        except (ValueError, TypeError):
            print(f"{key:<15}: {value}")
    print("---------------------------\n")

    return paths
