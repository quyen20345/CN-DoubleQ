#!/bin/bash
# scripts/run_extract.sh

# Script để chạy tác vụ trích xuất, chunking và indexing dữ liệu từ PDF.
# Cách dùng:
#   bash scripts/run_extract.sh public
#   bash scripts/run_extract.sh private
#   bash scripts/run_extract.sh training

# Lấy mode từ đối số dòng lệnh, nếu không có thì mặc định là 'public'
MODE=${1:-public}

echo "================================================="
echo "      BẮT ĐẦU CHẠY TÁC VỤ TRÍCH XUẤT (EXTRACT)"
echo "      CHẾ ĐỘ: $MODE"
echo "================================================="

# Di chuyển lên thư mục gốc của dự án để đảm bảo python path đúng
cd "$(dirname "$0")/.."

# Kiểm tra xem main.py có tồn tại không
if [ ! -f "main.py" ]; then
    echo "Lỗi: Không tìm thấy file 'main.py' ở thư mục gốc."
    exit 1
fi

# Chạy lệnh python cho tác vụ 'extract'
python main.py --mode "$MODE" --task extract

echo "================================================="
echo "      TÁC VỤ TRÍCH XUẤT CHO MODE '$MODE' ĐÃ HOÀN TẤT!"
echo "================================================="