#!/bin/bash
# Script để chạy tác vụ Trả lời câu hỏi (QA).
# Tác vụ này yêu cầu đã chạy tác vụ 'extract' trước đó.
# Mặc định chạy với mode 'public'. Thay đổi --mode thành 'training' hoặc 'private' nếu cần.

echo "--- Bắt đầu tác vụ Trả lời câu hỏi (QA) ---"
python3 -m main.src.main --mode public --task qa
# python3 -m main.src.main --mode private --task qa
echo "--- Hoàn thành tác vụ QA ---"
