#!/bin/bash
# scripts/prepare_data.sh

# Script này sẽ tự động giải nén các file .zip trong thư mục 'data'
# để tạo ra cấu trúc thư mục mà chương trình chính (main.py) yêu cầu.

# --- Cấu hình ---
# Di chuyển lên thư mục gốc của dự án để script có thể chạy từ bất cứ đâu
cd "$(dirname "$0")/.."
DATA_DIR="data"

echo "=========================================="
echo "    BẮT ĐẦU CHUẨN BỊ THƯ MỤC DỮ LIỆU"
echo "  Thư mục dữ liệu gốc: $(pwd)/$DATA_DIR"
echo "=========================================="
echo ""

# Kiểm tra thư mục data
if [ ! -d "$DATA_DIR" ]; then
    echo "Lỗi: Không tìm thấy thư mục '$DATA_DIR'. Vui lòng tạo thư mục này và đặt các file .zip vào trong."
    exit 1
fi

# Hàm xử lý giải nén chung
unzip_data() {
    local zip_file=$1
    local target_dir=$2

    if [ -f "$zip_file" ]; then
        echo "Tìm thấy file: $zip_file"
        echo "-> Đang giải nén vào: $target_dir"
        
        # Tạo thư mục đích và giải nén, ghi đè nếu cần
        mkdir -p "$target_dir"
        unzip -qo "$zip_file" -d "$target_dir" # -q: quiet, -o: overwrite
        
        echo "-> Giải nén hoàn tất."
    else
        echo "Cảnh báo: Không tìm thấy file $zip_file. Bỏ qua."
    fi
    echo "------------------------------------------"
}

# Xử lý các bộ dữ liệu
unzip_data "$DATA_DIR/public-test-input.zip" "$DATA_DIR/public_test_input"
unzip_data "$DATA_DIR/private-test-input.zip" "$DATA_DIR/private_test_input"
unzip_data "$DATA_DIR/training_input.zip" "$DATA_DIR/training_input"

echo "=========================================="
echo "    HOÀN TẤT CHUẨN BỊ DỮ LIỆU!"
echo "=========================================="
echo "Cấu trúc thư mục của bạn bây giờ đã sẵn sàng."
echo "Bạn có thể chạy pipeline bằng lệnh: python main.py --mode <mode> --task <task>"
