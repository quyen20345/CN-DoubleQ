#!/bin/bash
# scripts/run_choose_answer.sh

# Script để chạy tác vụ trả lời câu hỏi (QA).
# Yêu cầu: Phải chạy 'run_extract.sh' trước đó.
# Cách dùng:
#   bash scripts/run_choose_answer.sh public
#   bash scripts/run_choose_answer.sh private
#   bash scripts/run_choose_answer.sh training

# Lấy mode từ đối số dòng lệnh, nếu không có thì mặc định là 'public'
MODE=${1:-public}

echo "================================================="
echo "    BẮT ĐẦU CHẠY TÁC VỤ TRẢ LỜI CÂU HỎI (QA)"
echo "    CHẾ ĐỘ: $MODE"
echo "================================================="

# Di chuyển lên thư mục gốc của dự án để đảm bảo python path đúng
cd "$(dirname "$0")/.."

# Kiểm tra xem main.py có tồn tại không
if [ ! -f "main.py" ]; then
    echo "Lỗi: Không tìm thấy file 'main.py' ở thư mục gốc."
    exit 1
fi

# Chạy lệnh python cho tác vụ 'qa'
python main.py --mode "$MODE" --task qa

echo "================================================="
echo "    TÁC VỤ QA CHO MODE '$MODE' ĐÃ HOÀN TẤT!"
echo "    Kiểm tra kết quả trong thư mục: output/${MODE}_test_output/"
echo "================================================="
