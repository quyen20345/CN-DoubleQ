#!/bin/bash

# Script này sẽ tự động giải nén các file .zip trong thư mục 'data'
# để tạo ra cấu trúc thư mục mà chương trình chính (main.py) yêu cầu.

# --- Cấu hình ---
DATA_DIR="main/data"

# --- Hàm xử lý ---
unzip_data() {
    local zip_file=$1
    local target_dir=$2

    if [ -f "$zip_file" ]; then
        echo "Tìm thấy file: $zip_file"
        echo "-> Đang giải nén vào thư mục: $target_dir"
        
        # Tạo thư mục đích nếu chưa có
        mkdir -p "$target_dir"
        
        # Giải nén và ghi đè nếu file đã tồn tại
        unzip -o "$zip_file" -d "$target_dir"
        
        echo "-> Giải nén hoàn tất."
        echo ""
    else
        echo "Cảnh báo: Không tìm thấy file $zip_file. Bỏ qua."
        echo ""
    fi
}

# --- Bắt đầu thực thi ---
echo "=========================================="
echo "    BẮT ĐẦU CHUẨN BỊ THƯ MỤC DỮ LIỆU"
echo "=========================================="
echo ""

# Kiểm tra thư mục data
if [ ! -d "$DATA_DIR" ]; then
    echo "Lỗi: Không tìm thấy thư mục '$DATA_DIR'. Vui lòng tạo thư mục này và đặt các file .zip vào trong."
    exit 1
fi

# Xử lý public test data
unzip_data "$DATA_DIR/public-test-input.zip" "$DATA_DIR/public_test_input"

# Xử lý private test data
unzip_data "$DATA_DIR/private-test-input.zip" "$DATA_DIR/private_test_input"

# Xử lý training data
unzip_data "$DATA_DIR/training_input.zip" "$DATA_DIR/training_test_input"

# (Tùy chọn) Xử lý private test data nếu có
if [ -f "$DATA_DIR/private_test_input.zip" ]; then
    unzip_data "$DATA_DIR/private_test_input.zip" "$DATA_DIR/private_test_input"
fi


echo "=========================================="
echo "    HOÀN TẤT CHUẨN BỊ DỮ LIỆU!"
echo "=========================================="
echo "Cấu trúc thư mục của bạn bây giờ đã sẵn sàng."
echo "Bạn có thể chạy 'bash run_extract.sh' để bắt đầu."