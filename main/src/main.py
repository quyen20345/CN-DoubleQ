# main/src/main.py
# -*- coding: utf-8 -*-
"""
Entry point cho Nhiệm vụ 2: Khai phá tri thức từ văn bản kỹ thuật.
"""

import argparse
from main.src.core.config import setup_paths
from main.src.pipeline.task_extract import run_task_extract
from main.src.pipeline.task_qa import run_task_qa


def main():
    parser = argparse.ArgumentParser(description="Pipeline cho Nhiệm vụ 2 - Zalo AI Challenge")
    parser.add_argument(
        "--mode", choices=["public", "private", "training"], default="public",
        help="Chế độ chạy: public, private hoặc training."
    )
    parser.add_argument(
        "--task", choices=["extract", "qa", "full"], default="full",
        help="Tác vụ cần thực hiện: extract, qa, hoặc full."
    )
    args = parser.parse_args()

    print(f"\n{'*'*80}\n{' BẮT ĐẦU PIPELINE '.center(80,'*')}\n{'*'*80}")
    paths = setup_paths(args.mode)

    if args.task == "extract":
        run_task_extract(paths)
    elif args.task == "qa":
        run_task_qa(paths)
    elif args.task == "full":
        if run_task_extract(paths):
            run_task_qa(paths)

    print(f"\n{'*'*80}\n{' PIPELINE KẾT THÚC '.center(80,'*')}\n{'*'*80}\n")


if __name__ == "__main__":
    main()