"""
================================================================================
BÁO CÁO THỰC HIỆN LAB 6: GIỚI THIỆU VỀ TRANSFORMERS
================================================================================

[cite_start]Lab này nhằm mục tiêu ôn lại kiến trúc Transformer, sử dụng các mô hình tiền huấn luyện (pretrained models) để thực hiện các tác vụ NLP cơ bản như MLM và Text Generation, và làm quen với thư viện Hugging Face 'transformers'. [cite: 410, 412, 413]

1. CÁC BƯỚC TRIỂN KHAI (Implementation Steps)

[cite_start]A. Bài 1: Khôi phục Masked Token (MLM - Encoder-only) [cite: 443]
- [cite_start]Mô hình: Pipeline 'fill-mask' (mặc định tải mô hình họ RoBERTa/DistilBERT). [cite: 448, 449]
- [cite_start]Đầu vào: "Hanoi is the <mask> of Vietnam." [cite: 452]
- [cite_start]Cơ chế: Mô hình Encoder-only (như BERT) phù hợp vì chúng nhìn **hai chiều (bidirectional)**, hiểu toàn bộ ngữ cảnh xung quanh từ bị che để dự đoán. [cite: 426, 427, 462]

[cite_start]B. Bài 2: Dự đoán từ tiếp theo (Next Token Prediction - NTP - Decoder-only) [cite: 463]
- [cite_start]Mô hình: Pipeline 'text-generation' (mặc định tải mô hình họ GPT-2). [cite: 469, 470]
- [cite_start]Đầu vào: Câu mồi "The best thing about learning NLP is". [cite: 472]
- [cite_start]Cơ chế: Mô hình Decoder-only (như GPT) phù hợp vì chúng chỉ nhìn **một chiều (unidirectional)**, chuyên dự đoán token tiếp theo trong chuỗi. [cite: 432, 433, 482]

[cite_start]C. Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation) [cite: 483]
- [cite_start]Mô hình: BERT (bert-base-uncased). [cite: 496]
- [cite_start]Phương pháp: **Mean Pooling** (Lấy trung bình cộng các vector đầu ra, loại trừ token padding). [cite: 487, 490]

--------------------------------------------------------------------------------

2. CÁCH CHẠY CODE VÀ KẾT QUẢ LOG (How to run the code and log the results)

Mã nguồn được thực thi trong 5 cell trên Google Colab.

* [cite_start]**Kết quả Bài 1 (MLM):** Dự đoán hàng đầu là 'capital' với độ tin cậy cao nhất. [cite: 461]
    - Ví dụ: Dự đoán: 'capital' với độ tin cậy: 0.9850 -> Câu hoàn chỉnh: Hanoi is the capital of Vietnam.
* **Kết quả Bài 3 (Vector Biểu diễn):**
    - [cite_start]Kích thước (chiều) của vector biểu diễn: **torch.Size([1, 768])**. [cite: 525]

--------------------------------------------------------------------------------

3. GIẢI THÍCH KẾT QUẢ THU ĐƯỢC (Explain the obtained results)

- **Kích thước vector 768:** Con số này tương ứng với tham số **Hidden Size ($D_{model}$)** của mô hình `bert-base-uncased`. [cite_start]Đây là chiều dài cố định của vector biểu diễn cho mỗi token đầu ra và cũng là chiều dài của vector đại diện cho cả câu sau Mean Pooling. [cite: 527, 528]
- [cite_start]**Sự hợp lý của Văn bản sinh ra (Bài 2):** Các chuỗi được sinh ra từ GPT-2 thường rất hợp lý về mặt ngữ pháp và ngữ nghĩa, do mô hình được huấn luyện để tạo ra chuỗi tiếp theo có tính liên tục cao. [cite: 481]
- **Vai trò của Attention Mask (Bài 3):** Cần sử dụng `attention_mask` khi thực hiện Mean Pooling để **bỏ qua các vector của token đệm (padding tokens)**. [cite_start]Việc này giúp đảm bảo vector cuối cùng chỉ là trung bình của các token ngữ nghĩa thực tế trong câu, tránh làm sai lệch kết quả. [cite: 517, 529]

--------------------------------------------------------------------------------

4. KHÓ KHĂN GẶP PHẢI VÀ CÁCH KHẮC PHỤC (Difficulties encountered and how to solve them)

- **Khó khăn:** Lỗi `PipelineException: No mask_token (<mask>) found on the input` xảy ra ở Bài 1.
- **Nguyên nhân:** Mô hình mặc định tải bởi pipeline (`distilbert/distilroberta-base`) sử dụng token che là `<mask>`, nhưng câu đầu vào lại sử dụng `[MASK]` (token của BERT).
- **Khắc phục:** Thay đổi trực tiếp token che trong câu đầu vào từ `[MASK]` thành **`<mask>`** để phù hợp với yêu cầu tokenizer của mô hình.

5. THAM CHIẾU NGUỒN BÊN NGOÀI (If referencing external sources, clearly state the references)
- Thư viện chính: **Hugging Face Transformers** (được sử dụng cho tất cả các tác vụ pipeline và mô hình BERT).
- [cite_start]Nguồn kiến trúc: Bài báo **"Attention Is All You Need"** (Kiến trúc Transformer). [cite: 416]

6. SỬ DỤNG MÔ HÌNH TIỀN HUẤN LUYỆN (If using pre-trained models, clearly state which model, from where, and the prompt)
- Bài 1 (MLM): Mô hình **DistilRoBERTa-base** (mặc định) hoặc một biến thể BERT từ Hugging Face Hub.
- Bài 2 (NTP): Mô hình **GPT-2** (mặc định) từ Hugging Face Hub.
- [cite_start]Bài 3 (Vector): Mô hình **BERT-base-uncased** từ Hugging Face Hub. [cite: 496]
"""
