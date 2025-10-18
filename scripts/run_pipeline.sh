#!/bin/bash
# scripts/run_pipeline.sh

# Script tiện ích để chạy toàn bộ pipeline cho một mode cụ thể.
# Mặc định chạy ở chế độ 'public'.
# Cách dùng:
#   bash scripts/run_pipeline.sh public
#   bash scripts/run_pipeline.sh private
#   bash scripts/run_pipeline.sh training

# Lấy mode từ đối số dòng lệnh, nếu không có thì mặc định là 'public'
MODE=${1:-public}

echo "================================================="
echo "  BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE (FULL TASK)"
echo "  CHẾ ĐỘ: $MODE"
echo "================================================="

# Di chuyển lên thư mục gốc của dự án
cd "$(dirname "$0")/.."

# Kiểm tra xem main.py có tồn tại không
if [ ! -f "main.py" ]; then
    echo "Lỗi: Không tìm thấy file 'main.py' ở thư mục gốc. Vui lòng kiểm tra lại cấu trúc dự án."
    exit 1
fi

# Chạy lệnh python
python main.py --mode "$MODE" --task full

echo "================================================="
echo "  PIPELINE CHO MODE '$MODE' ĐÃ HOÀN TẤT!"
echo "  Kiểm tra kết quả trong thư mục: output/${MODE}_test_output/"
echo "================================================="
