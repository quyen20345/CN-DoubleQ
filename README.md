# CN-DoubleQ

cd ~/Documents/CN-DoubleQ
python3 -m main.src.main


Dữ liệu:

training_out.zip

public_test_input.zip

training_input.zip

TÁC VỤ 2: KHAI PHÁ TRI THỨC TỪ VĂN BẢN KỸ THUẬT

Trong các lĩnh vực công nghiệp, năng lượng, hàng không, hay y sinh, hàng triệu trang tài liệu kỹ thuật đang được tạo ra và lưu trữ dưới định dạng PDF mỗi ngày. Bên trong đó là vô vàn bảng biểu phức tạp: bảng trải dài hàng trăm trang, ô gộp chồng chéo, tiêu đề ngang dọc, ký hiệu toán học, thuật ngữ chuyên ngành song song cả tiếng Việt lẫn tiếng Anh. Những kho dữ liệu này chính là mỏ vàng tri thức, nhưng hiện nay phần lớn vẫn "nằm yên" dưới dạng văn bản khó truy cập.

Câu hỏi đặt ra: Làm thế nào để máy tính có thể tự động đọc, hiểu và trả lời truy vấn từ những bảng dữ liệu kỹ thuật khổng lồ này?

Trong bối cảnh đó, hai nhu cầu quan trọng được xác định:

Trích xuất dữ liệu: Chuyển đổi những bảng PDF phức tạp thành cấu trúc dữ liệu số chính xác, có thể khai thác và lưu trữ.

Truy vấn dữ liệu: Dựa trên dữ liệu văn bản đã được chuyển đổi, thí sinh thực hiện nhiệm vụ trả lời các câu hỏi trắc nghiệm có thể có nhiều đáp án đúng (Multiple Choice Q&A).

Nhiệm vụ

Thí sinh sẽ thực hiện hai nhiệm vụ mang tính nền tảng:

Nhiệm vụ 1: Trích xuất dữ liệu

Biến các bảng dữ liệu PDF nhiều tầng lớp thành dữ liệu số chuẩn lưu dưới dạng .md, có thể dùng ngay cho phân tích. Các phương pháp trích xuất hướng tới giải quyết các thách thức từ dữ liệu PDF với các đặc trưng như sau:

Chứa watermark phức tạp.

Bảng dài hàng trăm trang, có thể trải qua nhiều trang.

Merge cells ngang/dọc, tiêu đề lồng nhau.

Nội dung pha trộn đa ngôn ngữ, ký hiệu toán học và thuật ngữ chuyên ngành.

Yêu cầu chi tiết:

Mô hình trích xuất làm việc với tệp dữ liệu đầu vào định dạng .pdf.

Kết quả trích xuất được lưu dưới dạng tệp .md. Yêu cầu định dạng cụ thể như sau:

Các bảng được chuyển đổi sang HTML table trong Markdown.

Hình ảnh và công thức được thay thế bằng placeholder: |<image_n>|, |<formula_n>| theo thứ tự xuất hiện. Các hình ảnh và công thức sẽ được lưu cùng tệp .md.

Các thành phần khác (heading, bullet list, code block) phải giữ nguyên định dạng Markdown.

Nhiệm vụ 2: Truy vấn dữ liệu

Trả lời các câu hỏi trắc nghiệm với 4 lựa chọn A, B, C, D, trong đó mỗi câu có thể có nhiều đáp án đúng. Thí sinh phải dùng dữ liệu trích xuất được từ nhiệm vụ 1 để trả lời cho các câu hỏi trắc nghiệm đưa ra.

Yêu cầu chi tiết:

Các câu hỏi được lưu trong tệp question.csv chứa các câu hỏi. Mỗi câu đi kèm 4 lựa chọn A, B, C, D.

Mỗi câu hỏi cần bóc ra số lượng câu hỏi trả lời đúng và danh sách các câu trả lời.

Cấu trúc dữ liệu:

Training Data

Input: Một tệp training_input.zip gồm các tệp PDF cần trích xuất dữ liệu và một tệp question.csv. Tệp question.csv có định dạng như sau:

Cột đầu tiên là câu hỏi.

Bốn cột tiếp theo tương ứng với bốn đáp án A, B, C, D.

Output: Một tệp training_output.zip gồm một tệp answer.md và một tập các thư mục con.

Tên thư mục con là tên của tệp PDF cần trích xuất. Trong mỗi thư mục con sẽ có:

Tệp main.md chứa nội dung trích xuất.

Thư mục con images chứa ảnh và công thức được trích xuất.

Tệp answer.md là kết quả tổng hợp của cả 2 nhiệm vụ. Nội dung tệp gồm hai phần:

Phần trích xuất: bắt đầu từ ### TASK EXTRACT, tiếp theo là # tên_tệp_pdf và nội dung được trích xuất (trùng với nội dung trong tệp main.md).

Phần QA: bắt đầu từ ### TASK QA, sau đó là thông tin gồm số lượng câu đúng và danh sách các đáp án đúng. Thứ tự các câu trả lời trắc nghiệm từ trên xuống được giữ nguyên như thứ tự các câu hỏi trắc nghiệm trong tệp training_question.csv.

Public Test Data: Thí sinh sẽ được cung cấp tệp public_test_input.zip có cấu trúc như tệp training_input.zip. Thí sinh phải nộp tệp public_test_output.zip có cấu trúc như tệp training_output.zip.

Private Test Data: Thí sinh sẽ được cung cấp tệp private_test_input.zip có cấu trúc như tệp training_input.zip. Thí sinh phải nộp tệp private_test_output.zip có cấu trúc như tệp training_output.zip.

Chú ý: Thí sinh xem định dạng mẫu từ tập training.

Nộp bài: Ngoài tệp answer.md và các thư mục con trích xuất từ các PDF, thí sinh cần nộp tệp main.py được đặt trong cùng file nén public_test_output.zip và private_test_output.zip. Tệp main.py chứa tất cả source code trong project của thí sinh. Chú ý: Tệp main.py này không yêu cầu phải chạy được, mà giúp ban giám khảo kiểm tra trước tính trung thực của nhóm.

Yêu cầu kỹ thuật

Bắt buộc sử dụng các mô hình mã nguồn mở với số lượng tham số dưới 4B cho mỗi nhiệm vụ.

Thí sinh được phép áp dụng các phương pháp data augmentation.

Sau mỗi giai đoạn, yêu cầu mỗi đội cung cấp đường dẫn (link) chứa toàn bộ mã nguồn, tệp requirement, checkpoint và hướng dẫn huấn luyện để tiến hành hậu kiểm và đánh giá mô hình. Thí sinh cần nộp đường dẫn GitHub (ở chế độ chỉ chia sẻ cho BTC). Hệ thống nộp phải là một pipeline thống nhất:

Khi chạy tệp run_extract.sh, hệ thống phải tự động chuyển đổi dữ liệu đầu vào dạng PDF thành tập dữ liệu Markdown giống với kết quả nộp. Dữ liệu cần được xử lý và lập chỉ mục (index) hoàn toàn trên môi trường cục bộ (RAM hoặc file local), không được sử dụng cơ sở dữ liệu bên ngoài.

Khi chạy tệp run_choose_answer.sh, hệ thống phải tự động duyệt qua các câu hỏi do BTC cung cấp và xuất ra tệp kết quả tương tự kết quả nộp.

│ ├── data

│ │ ├── public_test_input

│ │ │ └── public-test-input

│ │ │ ├── Public061.pdf

│ │ │ ├── Public062.pdf

│ │ │ ├── Public063.pdf

│ │ │ ├── Public064.pdf

│ │ │ ├── Public065.pdf

│ │ │ ├── Public066.pdf

│ │ │ ├── Public067.pdf

│ │ │ ├── Public068.pdf

│ │ │ ├── Public069.pdf

│ │ │ ├── Public070.pdf

│ │ │ ├── Public071.pdf

│ │ │ ├── Public072.pdf

│ │ │ ├── Public073.pdf

│ │ │ ├── Public074.pdf

│ │ │ ├── Public075.pdf

│ │ │ ├── Public076.pdf

│ │ │ ├── Public077.pdf

│ │ │ ├── Public078.pdf

│ │ │ ├── Public079.pdf

│ │ │ ├── Public080.pdf

│ │ │ └── question.csv

│ │ ├── public-test-input.zip

│ │ ├── training_input.zip

│ │ ├── training_out

│ │ │ ├── Public001

│ │ │ │ ├── images

│ │ │ │ └── main.md

│ │ │ └── Public002

│ │ │ ├── images

│ │ │ └── main.md

│ │ ├── training_out.zip

│ │ └── training_test_input

│ │ └── training_input

│ │ ├── Public001.pdf

│ │ ├── Public002.pdf

│ │ ├── Public003.pdf

│ │ ├── Public004.pdf

│ │ ├── Public005.pdf

│ │ ├── Public006.pdf

│ │ ├── Public007.pdf

│ │ ├── Public008.pdf

│ │ ├── Public009.pdf

│ │ ├── Public010.pdf

│ │ ├── Public011.pdf

│ │ ├── Public012.pdf

│ │ ├── Public013.pdf

│ │ ├── Public014.pdf

│ │ ├── Public015.pdf

│ │ ├── Public016.pdf

│ │ ├── Public017.pdf

│ │ ├── Public018.pdf

│ │ ├── Public019.pdf

│ │ ├── Public020.pdf

│ │ ├── Public021.pdf

│ │ ├── Public022.pdf

│ │ ├── Public023.pdf

│ │ ├── Public024.pdf

│ │ ├── Public025.pdf

│ │ ├── Public026.pdf

│ │ ├── Public027.pdf

│ │ ├── Public028.pdf

│ │ ├── Public029.pdf

│ │ ├── Public030.pdf

│ │ ├── Public031.pdf

│ │ ├── Public032.pdf

│ │ ├── Public033.pdf

│ │ ├── Public034.pdf

│ │ ├── Public035.pdf

│ │ ├── Public036.pdf

│ │ ├── Public037.pdf

│ │ ├── Public038.pdf

│ │ ├── Public039.pdf

│ │ ├── Public040.pdf

│ │ ├── Public041.pdf

│ │ ├── Public042.pdf

│ │ ├── Public043.pdf

│ │ ├── Public044.pdf

│ │ ├── Public045.pdf

│ │ ├── Public046.pdf

│ │ ├── Public047.pdf

│ │ ├── Public048.pdf

│ │ ├── Public049.pdf

│ │ ├── Public050.pdf

│ │ ├── Public051.pdf

│ │ ├── Public052.pdf

│ │ ├── Public053.pdf

│ │ ├── Public054.pdf

│ │ ├── Public055.pdf

│ │ ├── Public056.pdf

│ │ ├── Public057.pdf

│ │ ├── Public058.pdf

│ │ ├── Public059.pdf

│ │ ├── Public060.pdf

│ │ └── question.csv

├── docker-compose.yaml

├── LICENSE

├── main

│ ├── configs

│ │ └── config.yaml

│ ├── data

│ ├── __init__.py

│ ├── output

│ │ └── output.txt

│ ├── __pycache__

│ │ └── __init__.cpython-310.pyc

│ └── src

│ ├── data

│ ├── embedding

│ ├── embedding_models

│ ├── __init__.py

│ ├── llm

│ ├── main.py

│ ├── output

│ ├── __pycache__

│ ├── utils

│ └── vectordb

├── prepare_data.sh

├── README.md

├── requirements.txt

├── run_choose_answer.sh

└── run_extract.sh