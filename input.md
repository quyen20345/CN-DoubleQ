Tiêu đề cấp 1: Tổng quan về MarkdownMarkdown là một ngôn ngữ đánh dấu nhẹ được tạo ra bởi John Gruber vào năm 2004. Mục tiêu của nó là cho phép mọi người "viết bằng định dạng văn bản thuần túy dễ đọc, dễ viết" và tùy chọn chuyển đổi nó thành HTML (và nhiều định dạng khác).Tiêu đề cấp 2: Các yếu tố định dạng cơ bảnĐây là một số định dạng văn bản phổ biến nhất.Chữ in đậm để nhấn mạnh.Chữ in nghiêng cho các thuật ngữ hoặc sự nhấn mạnh nhẹ nhàng.Vừa đậm vừa nghiêng cho sự nhấn mạnh tối đa.Mã được dùng để hiển thị các biến hoặc đoạn mã ngắn.~~Chữ gạch ngang~~ để biểu thị nội dung đã bị xóa hoặc không còn phù hợp.Tiêu đề cấp 3: Danh sáchBạn có thể tạo nhiều loại danh sách khác nhau.Danh sách có thứ tự:Mục con thứ nhất.Mục con thứ hai.Danh sách không thứ tự lồng nhau.Có thể chứa mã bên trong.Mục con thứ ba.Danh sách không thứ tự:Sử dụng dấu hoa thị.Hoặc dấu gạch ngang.Hoặc dấu cộng.Tiêu đề cấp 4: Danh sách công việc (Task Lists)[x] Tác vụ đã hoàn thành.[ ] Tác vụ chưa hoàn thành.[ ] Một tác vụ khác cần làm, có thể chứa định dạng.Tiêu đề cấp 2: Các yếu tố nâng caoTrích dẫn (Blockquotes)Sử dụng trích dẫn để làm nổi bật văn bản từ các nguồn khác."Sự khác biệt duy nhất giữa một ngày tốt và một ngày tồi tệ là thái độ của bạn."Trích dẫn lồng nhau.Và có thể lồng sâu hơn nữa.Khối mã (Code Blocks)Đây là một ví dụ về một khối mã Python với tô sáng cú pháp.def semantic_chunker(text, model, threshold):
    """
    Splits a text into semantically coherent chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return []

    embeddings = model.encode(sentences)
    print("Embeddings created successfully.")

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(sentences) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        if sim > threshold:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]

    chunks.append(" ".join(current_chunk))
    return chunks
Bảng (Tables)Bảng là một cách tuyệt vời để tổ chức dữ liệu.Căn tráiCăn giữaCăn phảiDữ liệu 1Dữ liệu 2Dữ liệu 3Nội dungVui vẻ$1600Z-AA-Z1-100Liên kết và Hình ảnhBạn có thể liên kết đến các trang web khác hoặc hiển thị hình ảnh.Đây là một liên kết đến Google.Bạn cũng có thể sử dụng liên kết tham chiếu.Đây là một hình ảnh placeholder:Tiêu đề cấp 5: Ghi chú thêmĐây chỉ là một ví dụ nhỏ về những gì Markdown có thể làm. Nó rất linh hoạt và mạnh mẽ. 🚀Tiêu đề cấp 6: Kết luậnHy vọng tệp này hữu ích cho việc kiểm thử của bạn!