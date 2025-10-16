## note1
### host model lên kaggle để tận dụng gpu tăng tốc độ xử lý:
- có vi phạm rule mà viettel đưa ra không?

### Sau khi test với code ở round1 thì thực hiện:
- mở question mà ban tổ chức cho sau đó dùng tìm kết quả đúng.
- lấy answer.md được sinh hiện tại check đáp án nào sai, đáp án nào đúng.
- phân tích các intent tìm hướng giải quyết.

### Tối ưu lại đường dẫn cho code.
- vì đường dẫn round1 và round2 khác nhau một xíu nên cần sửa.
- cách hiện tại là đổi tên file sau khi extract bằng tay như path của round1.

### Cảm tính và suy luận về các round
- round1 hỏi liên quan các câu hỏi xh dễ không cần reasoning 
- round2 liên quan học thuật 
---> round3 có thể là một mảng khác.
- Vậy cần cho qua một model để phân loại câu hỏi nếu câu hỏi thuộc intent nào thì sử dụng code hoặc model phù hợp với intent đó.

### suy luận thứ 2 về question.csv
- các câu trả lời có phải là được copy một đoạn trong pdf không? 
- Nếu có thì Thực hiện dùng một model dưới 4b sinh các cặp câu hỏi và câu trả lời để train model.

###  dữ liệu huấn luyện (bao gồm cả dữ liệu đã tăng cường - augmented data), tệp requirement, checkpoint, mô hình huấn luyện cho kết quả trùng khớp
- Suy nghĩ cách tăng cường dữ liệu.
- Suy nghĩ cách training model.


### Bắt buộc sử dụng các mô hình mã nguồn mở với số lượng tham số dưới 4B cho mỗi nhiệm vụ.


### Vấn đề liên quan đến llm.
- hiện tại đang fix temperature = 0
- nhưng tôi nghĩ nên cho temperature biến đổi phù hợp với từng dạng câu hỏi
- sử dụng llm với ollama thì có thể config những thông số nào nữa?

### Có nên phân nhỏ các file có lượng code lớn không?

### mày cũng cài mà chạy đi một mình tao làm ko nổi.

### check phần xử lý dữ liệu dạng bảng xem có extract đúng ko?
- sử dụng thử ocr cho những phần khó.

### hiên tại đang sử dụng model 3b
- có nên thử nghiệm 4b.

### đang embedding với model nào?
- có nên thay thế ko?