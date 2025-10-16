# CN-DoubleQ - Khai phá tri thức từ văn bản kỹ thuật

Dự án tham gia Zalo AI Challenge 2024 - Nhiệm vụ 2: Khai phá tri thức từ văn bản kỹ thuật.

## 🎯 Mô tả dự án

Hệ thống trích xuất và phân tích tài liệu PDF kỹ thuật, trả lời câu hỏi trắc nghiệm dựa trên nội dung đã trích xuất.

### Tính năng chính:
- **Trích xuất PDF**: Chuyển đổi tài liệu PDF thành Markdown với bảng, hình ảnh, công thức
- **Vector Search**: Tìm kiếm thông tin bằng embedding model
- **QA System**: Trả lời câu hỏi trắc nghiệm với độ chính xác cao
- **Multi-mode**: Hỗ trợ public, private, training data

## 🚀 Cài đặt

### Yêu cầu hệ thống:
- Python 3.8+
- RAM: 8GB+ (khuyến nghị 16GB)
- Disk: 10GB+ free space

### Cài đặt dependencies:

```bash
# Clone repository
git clone <repository-url>
cd CN-DoubleQ

# Cài đặt dependencies
pip install -r requirements.txt

# Khởi động Qdrant vector database
docker-compose up -d
```

## 🚀 Hướng dẫn chạy dự án

### Bước 1: Chuẩn bị môi trường

1. **Cài đặt Docker và Docker Compose** (nếu chưa có):
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Hoặc cài đặt Docker Desktop
```

2. **Clone repository và cài đặt dependencies**:
```bash
# Clone repository
git clone <repository-url>
cd CN-DoubleQ

# Tạo virtual environment (khuyến nghị)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu

1. **Đặt các file dữ liệu vào thư mục `main/data/`**:
   - `public-test-input.zip` → cho public test
   - `private-test-input.zip` → cho private test  
   - `training_input.zip` → cho training data

2. **Chạy script chuẩn bị dữ liệu**:
```bash
bash prepare_data.sh
```

### Bước 3: Khởi động services

**Khởi động Qdrant vector database và Ollama**:
```bash
docker-compose up -d
```

Kiểm tra services đang chạy:
```bash
docker-compose ps
```

### Bước 4: Chạy pipeline

#### 🎯 Chế độ Public Test
```bash
# Trích xuất PDF từ public test data
python3 main/src/main.py --mode public --task extract

# Trả lời câu hỏi
python3 main/src/main.py --mode public --task qa
```

#### 🔒 Chế độ Private Test  
```bash
# Trích xuất PDF từ private test data
python3 main/src/main.py --mode private --task extract

# Trả lời câu hỏi
python3 main/src/main.py --mode private --task qa
```

#### 📚 Chế độ Training
```bash
# Trích xuất PDF từ training data
python3 main/src/main.py --mode training --task extract

# Trả lời câu hỏi
python3 main/src/main.py --mode training --task qa
```

### Bước 5: Sử dụng Scripts tự động

**Sử dụng scripts có sẵn**:
```bash
# Trích xuất PDF (mặc định: private mode)
bash run_extract.sh

# Trả lời câu hỏi (mặc định: private mode)  
bash run_choose_answer.sh
```

**Chỉnh sửa mode trong scripts**:
- Mở `run_extract.sh` và `run_choose_answer.sh`
- Thay đổi `--mode private` thành `--mode public` hoặc `--mode training`

## 📊 Kết quả

Sau khi chạy xong, kết quả sẽ được lưu trong:
- `output/public_test_output/` - Kết quả public test
- `output/private_test_output/` - Kết quả private test  
- `output/training_test_output/` - Kết quả training

Mỗi thư mục chứa:
- `images/` - Hình ảnh trích xuất từ PDF
- `main.md` - Nội dung markdown đã xử lý
- Các file khác theo cấu trúc của từng PDF

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Lỗi "Module not found"**:
```bash
# Đảm bảo đang ở thư mục gốc của project
cd /path/to/CN-DoubleQ
python3 main/src/main.py --mode public --task extract
```

2. **Lỗi "Connection refused" với Qdrant**:
```bash
# Kiểm tra Docker services
docker-compose ps
docker-compose logs qdrant

# Restart services nếu cần
docker-compose restart
```

3. **Lỗi "File not found" cho question.csv**:
```bash
# Chạy lại script chuẩn bị dữ liệu
bash prepare_data.sh
```

4. **Lỗi memory/GPU**:
```bash
# Giảm batch size trong code hoặc tăng RAM
# Kiểm tra GPU memory nếu sử dụng CUDA
```

### Kiểm tra logs:
```bash
# Xem logs của Docker services
docker-compose logs -f

# Xem logs của Python script
python3 main/src/main.py --mode public --task extract 2>&1 | tee logs.txt
```

## 🛠️ Cấu hình nâng cao

### Thay đổi model embedding:
Chỉnh sửa trong `main/src/embedding/model.py`

### Thay đổi LLM model:
Chỉnh sửa trong `main/src/llm/llm_integrations.py`

### Tùy chỉnh vector database:
Chỉnh sửa trong `main/src/vectordb/qdrant.py`

## 📝 Ghi chú

- **RAM**: Dự án yêu cầu ít nhất 8GB RAM, khuyến nghị 16GB+
- **Disk**: Cần ít nhất 10GB free space cho models và data
- **GPU**: Không bắt buộc nhưng sẽ tăng tốc độ xử lý
- **Internet**: Cần kết nối internet để download models lần đầu
