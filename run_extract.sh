#!/bin/bash
# Script để chạy tác vụ trích xuất PDF.
# Mặc định chạy với mode 'public'. Thay đổi --mode thành 'training' hoặc 'private' nếu cần.

echo "--- Bắt đầu tác vụ trích xuất PDF ---"
python -m main.src.main --mode public --task extract
# python -m main.src.main --mode private --task extract
echo "--- Hoàn thành tác vụ trích xuất ---"