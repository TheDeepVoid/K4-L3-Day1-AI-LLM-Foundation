# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Qua bốn phản hồi, tất cả các mức temperature (từ 0.0 đến 1.5) đều hội tụ về cùng một chủ đề cốt lõi là Hang Sơn Đoòng với các chi tiết đặc trưng giống nhau như kích thước chứa tòa nhà 40 tầng, hệ sinh thái và mây riêng. Điểm khác biệt duy nhất là khi temperature tăng lên, mức độ biến đổi từ vựng, văn phong và cấu trúc câu trở nên phong phú, linh hoạt hơn, chuyển từ đoạn văn sang danh sách gạch đầu dòng và thêm các chi tiết diễn đạt sáng tạo.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Từ 0 tới 0.2. Vì chatbot duy trì tính nhất quán, độ chính xác cao và giảm thiểu hiện tượng hallucination. Trong hoạt động hỗ trợ khách hàng, ưu tiên hàng đầu là cung cấp đúng chính sách, quy trình và thông tin sản phẩm thay vì sự sáng tạo hay biến đổi văn phong.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Tổng số lượt gọi (requests): $10.000 \text{ người} \times 3 \text{ lần} = 30.000 \text{ requests/ngày}$.Tổng token đầu ra (output): $30.000 \times 350 = 10.500.000 \text{ tokens/ngày} = 10.500 \text{ nghìn tokens/ngày}$.
>
> Đơn giá:
>
> * GPT-4o Input: 0.0025 / GPT-4o-mini Input: 0.00015 -> Gấp 16,67 lần<sup></sup>
> * GPT-4o Output: 0.0100/ GPT-4o-mini Output: 0.00060 -> Gấp 16,67 lần<sup></sup>
>
> Chi phí Token Output hàng ngày:<sup></sup>
>
> * GPT-4o:<sup></sup> $10.500 \times \$0.010 = \mathbf{\$105/\text{ngày}} \ (\approx \$3.150/\text{tháng})$
> * GPT-4o-mini:<sup></sup> $10.500 \times \$0.0006 = \mathbf{\$6,3/\text{ngày}} \ (\approx \$189/\text{tháng})$
>
>
> Trường hợp nên dùng GPT-4o là phân tích báo cáo tài chính phức tạp hoặc tư vấn pháp lý chuyên sâu, nơi yêu cầu khả năng suy luận logic nâng cao và độ chính xác tuyệt đối để tránh thiệt hại lớn.
>
> Trường hợp nên dùng GPT-4o-mini là chatbot phân loại ý định người dùng hoặc trích xuất thông tin định dạng JSON từ câu hỏi ngắn, nơi tác vụ đơn giản và việc dùng mô hình nhỏ giúp tiết kiệm hơn 94% chi phí mà vẫn đảm bảo hiệu năng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Độ dài & Cấu trúc: Cùng dung lượng (\~1.400 ký tự), bản giáo viên chia đoạn dài để kể chuyện, bản chuyên gia dùng mục lục và gạch đầu dòng kỹ thuật.
>
> * **Từ vựng:** Bản giáo viên dùng từ ngữ đời thường (*sổ tay, cái kẹo, bút xóa*); bản chuyên gia dùng thuật ngữ chuyên ngành (*DLT, Merkle Root, Nonce, mã băm*).
> * **Ví dụ:** Bản giáo viên ẩn dụ qua "cuốn sổ chung của lớp học", bản chuyên gia đi thẳng vào mô tả cấu trúc dữ liệu mà không dùng ví dụ đời sống.
>
> **Tác động của System Prompt:** System prompt định hình tư cách (persona) và tệp khách hàng mục tiêu, giúp mô hình tự động điều chỉnh tông giọng, độ sâu tri thức và cách đóng gói thông tin phù hợp mà không làm thay đổi bản chất sự thật

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Ước lượng theo công thức tiếng Anh (số từ / 0,75) bị sai lệch -17,1% so với tiktoken thực tế (165 so với 137 tokens). Do đặc thù tiếng Việt có cấu trúc âm tiết tách rời và hệ thống dấu thanh phức tạp, việc tính toán số lượng token bằng bộ mã hóa chuyên dụng (tiktoken) là bắt buộc để đảm bảo độ chính xác khi dự toán chi phí API.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực như chatbot hay công cụ viết lách, nơi giảm thời gian chờ phản hồi đầu tiên (Time to First Token - TTFT) giúp tạo cảm giác mượt mà và duy trì trải nghiệm người dùng không bị gián đoạn. Ngược lại, non-streaming lại phù hợp hơn đối với các tác vụ xử lý tự động ngầm (batch processing), lệnh gọi hàm/công cụ (function/tool calling), trích xuất dữ liệu định dạng JSON, hoặc khi ứng dụng cần phân tích toàn bộ văn bản phản hồi trước khi hiển thị/xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff tăng dần thời gian chờ theo cấp số nhân (\$1s \\rightarrow 2s \\rightarrow 4s \\dots\$), giúp giảm nhanh áp lực truy cập và cho phép hệ thống bị quá tải có đủ thời gian giải phóng tài nguyên để phục hồi.
>
> Nếu hàng nghìn client cùng retry với delay cố định (như 1 giây), toàn bộ request sẽ đồng loạt dội vào máy chủ tại cùng một thời điểm. Điều này tạo ra các đỉnh lưu lượng (traffic spikes) lặp đi lặp lại theo chu kỳ, gây ra hiện tượng **thảm họa cuộn tuyết (thundering herd problem)** khiến hệ thống tiếp tục nghẽn và có nguy cơ sụp đổ hoàn toàn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> 'Bạn là chuyên gia tư vấn thân thiện. Trả lời ngắn gọn, đúng trọng tâm bằng tiếng Việt; nếu không chắc chắn thì nói rõ thay vì bịa đặt.'
>
>
> * "trả lời ngắn gọn, đúng trọng tâm": Giới hạn token đầu ra (output token) để giảm chi phí API, tăng tốc độ phản hồi và giúp người dùng tiếp thu thông tin quan trọng ngay lập tức.
> * "nếu không chắc chắn thì nói rõ thay vì bịa đặt": Ngăn chặn hiện tượng bịa đặt thông tin (hallucination), đảm bảo độ tin cậy và tính an toàn cho dữ liệu tư vấn.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Trợ lý bị cắt cứng lịch sử hội thoại (`history = history[-6:]`) tối đa 3 lượt hỏi–đáp. Khi cuộc trò chuyện kéo dài, trợ lý sẽ hoàn toàn quên context, các yêu cầu hoặc thông tin quan trọng mà người dùng đã cung cấp ở các lượt đầu.
>
> Cải thiện: Triển khai Tóm tắt lịch sử tự động (Conversation Summary / Memory Buffer) thay vì cắt cứng.
>
> Cách triển khai:
>
> 1. Theo dõi kích thước: Khi len(history) > 6, không xóa bỏ các message cũ mà gửi các lượt hội thoại vượt quá ngưỡng này sang một hàm tóm tắt.
> 2. Tạo Summary: Gọi một API nhẹ (như gemini-2.5-flash hoặc gpt-4o-mini) để tóm tắt các lượt hội thoại cũ thành một đoạn văn ngắn lưu vào biến summary.
> 3. Cập nhật Messages Structure: Khi gửi request tiếp theo, chèn summary vào ngay sau system\_prompt để duy trì ngữ cảnh dài hạn với chi phí token tối thiểu.

**Đề xuất cải thiện:** Triển khai **Tóm tắt lịch sử tự động (Conversation Summary / Memory Buffer)** thay vì cắt cứng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [X]  `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [X]  Cả 4 checkpoint pytest đều pass
- [X]  Tất cả 9 câu trong file này đã được trả lời
- [X]  Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
