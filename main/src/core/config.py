# main/src/core/config.py
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_paths(mode: str) -> dict:
    """Thiết lập và xác thực các đường dẫn input và output."""
    if mode == "public":
        base_input_dir = project_root / f"data/{mode}_test_input/{mode}-test-input"
    elif mode == "private":
        base_input_dir = project_root / f"data/{mode}_test_input/{mode}_test_input"
    else:
        base_input_dir = project_root / f"data/{mode}_test_input/{mode}_input"


    output_dir = project_root / f"output/{mode}_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    question_csv = next(base_input_dir.rglob("question.csv"), None)
    if not question_csv:
        raise FileNotFoundError(f"Không tìm thấy 'question.csv' trong {base_input_dir}")

    paths = {
        "project_root": project_root,
        "pdf_dir": base_input_dir,
        "question_csv": question_csv,
        "output_dir": output_dir,
        "zip_name": f"{mode}_test_output.zip",
    }

    print("\n--- Cấu hình đường dẫn ---")
    for k, v in paths.items():
        try:
            print(f"{k:<15}: {Path(v).relative_to(project_root)}")
        except Exception:
            print(f"{k:<15}: {v}")
    print("---------------------------\n")

    return paths
