#!/bin/bash
# Script để chạy tác vụ Trả lời câu hỏi (QA).
# Tác vụ này yêu cầu đã chạy tác vụ 'extract' trước đó.
# Mặc định chạy với mode 'public'. Thay đổi --mode thành 'training' hoặc 'private' nếu cần.

echo "--- Bắt đầu tác vụ Trả lời câu hỏi (QA) ---"
python3 main/src/main.py --mode public --task qa
echo "--- Hoàn thành tác vụ QA ---"
