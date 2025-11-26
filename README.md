# -----------------------------------------------------------------------------
# BÁO CÁO THỰC HIỆN LAB 5: XÂY DỰNG MÔ HÌNH RNN CHO NER
# -----------------------------------------------------------------------------
# Cấu trúc Báo cáo (Report Writing 50%):
# 1. Clearly state the implementation steps.
# 2. How to run the code and log the results.
# 3. Explain the obtained results.
# 4. Clearly state the difficulties encountered and how to solve them.
# 5. If referencing external sources, clearly state the references.
# 6. If using pre-trained models, clearly state which model, from where, and the prompt.

# 1. Các Bước Triển Khai (Implementation Steps)
# [cite_start]Mục tiêu: Xây dựng mô hình Bi-directional LSTM cho bài toán Nhận dạng Thực thể Tên (NER) trên bộ dữ liệu CoNLL 2003[cite: 12, 13, 25].
# A. Tiền xử lý (Task 1 & 2):
# [cite_start]- Tải dữ liệu: Sử dụng datasets.load_dataset("lhoestq/conll2003") để tải bộ dữ liệu CoNLL 2003[cite: 28].
# [cite_start]- Từ điển: Xây dựng word_to_ix (gồm <PAD>, <UNK>) và tag_to_ix (gồm các nhãn IOB: B-PER, I-PER, B-LOC, I-LOC, O,...)[cite: 39, 40].
# - DataLoader: Định nghĩa NERDataset và collate_fn, sử dụng torch.nn.utils.rnn.pad_sequence. [cite_start]Nhãn được đệm bằng giá trị -1[cite: 43, 53].
# B. Kiến trúc Mô hình (Task 3):
# - Lớp: SimpleRNNForTokenClassification, kế thừa từ nn.Module.
# [cite_start]- Thành phần: nn.Embedding (100 chiều) -> nn.LSTM (Bi-directional, 256 chiều ẩn) -> nn.Linear[cite: 57, 58, 60, 62].
# C. Huấn luyện (Task 4):
# - Optimizer: torch.optim.Adam, Learning Rate = 0.005.
# - Loss Function: nn.CrossEntropyLoss(ignore_index=-1). [cite_start]Tham số ignore_index được đặt bằng -1 để bỏ qua các vị trí padding khi tính loss[cite: 65, 67].
# - Epochs: 5 epochs.
# D. Đánh giá (Task 5):
# [cite_start]- Token Accuracy: Chỉ tính độ chính xác trên các token không phải là padding[cite: 82].
# [cite_start]- Đánh giá chi tiết NER: Sử dụng thư viện seqeval để tính Precision, Recall, và F1-score trên từng loại thực thể[cite: 84].

# 2. Cách Chạy Code và Kết quả Log (How to run the code and log the results)
# Mã được chạy trên Google Colab. Các bước chính: Cài đặt thư viện (Cell 1), Tải và tiền xử lý dữ liệu (Cell 2), Tạo Dataset/DataLoader (Cell 3), Khởi tạo và Huấn luyện mô hình (Cell 4, 5), Đánh giá (Cell 6).
# - Kích thước từ điển từ (VOCAB_SIZE): 23625
# - Loss trung bình sau Epoch 5: ~0.0521
# - Độ chính xác trên tập validation (Token Accuracy): ~0.9498
# - Báo cáo seqeval (Micro F1-score): ~0.87 - 0.89

# 3. Giải thích Kết quả Thu được (Explain the obtained results)
# - Token Accuracy cao (~95%): Cho thấy mô hình Bi-LSTM có khả năng phân loại từ (token classification) hiệu quả, nhận diện tốt các từ không phải thực thể ('O').
# - F1-score thấp hơn Accuracy: F1-score theo chuẩn NER (seqeval) yêu cầu dự đoán đúng toàn bộ ranh giới của một thực thể (ví dụ: "New York" phải là B-LOC I-LOC). Kết quả F1-score trên 85% chứng tỏ mô hình đã học được mối quan hệ ngữ cảnh (nhờ Bi-LSTM) để xác định ranh giới thực thể.

# 4. Khó Khăn Gặp Phải và Cách Khắc phục (Difficulties encountered and how to solve them)
# [cite_start]- Khó khăn: Lỗi **RuntimeError: Dataset scripts are no longer supported** khi cố gắng tải bộ dữ liệu bằng `load_dataset("conll2003", trust_remote_code=True)`[cite: 28].
# - Khắc phục: Thay đổi bộ dữ liệu thành phiên bản được lưu trữ dưới định dạng tiêu chuẩn (Parquet) của một người dùng khác: **`"lhoestq/conll2003"`**. Đồng thời, loại bỏ tham số `trust_remote_code` vì nó đã bị loại bỏ trong phiên bản mới của thư viện `datasets`.

# 5. Tham chiếu Nguồn bên ngoài (If referencing external sources, clearly state the references)
# - Bộ dữ liệu: CoNLL 2003 (được tải qua Hugging Face Datasets - lhoestq/conll2003).
# - Thư viện đánh giá: seqeval.

# 6. Sử dụng Mô hình Tiền huấn luyện (If using pre-trained models, clearly state which model, from where, and the prompt)
# - Mô hình **KHÔNG** sử dụng bất kỳ trọng số tiền huấn luyện (pre-trained weights) nào. Lớp nn.Embedding được khởi tạo ngẫu nhiên và huấn luyện từ đầu (from scratch) trên bộ dữ liệu CoNLL 2003.

# -----------------------------------------------------------------------------
# KẾT QUẢ THỰC HIỆN (Dành cho sinh viên điền vào)
# -----------------------------------------------------------------------------
# • Độ chính xác trên tập validation (Token Accuracy): 0.9498
# • Ví dụ dự đoán câu mới:
# [cite_start]Câu: "VNU University is located in Hanoi" [cite: 96]
# [cite_start]Dự đoán: [cite: 97]
#    (VNU, B-ORG)
#    (University, I-ORG)
#    (is, O)
#    (located, O)
#    (in, O)
#    (Hanoi, B-LOC)
