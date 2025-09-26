# Báo cáo Dự án — Speech Emotion Recognition

**Sinh viên:** Nguyễn Đức Bình
**Môn:** Deep Learning — Lab 4
**Ngày nộp:** 2025-09-26

## 1. Mục tiêu

Tóm tắt mục tiêu bài toán SER và phạm vi (dataset, số lớp cảm xúc, ràng buộc).

## 2. Dữ liệu

- Bộ dữ liệu sử dụng: RAVDESS / EmoDB (mô tả nguồn, giấy phép, số lượng mẫu).
- Tiền xử lý: resample 16000 Hz, pad/trim 3.0s.
- Phân chia: train/val/test (tỷ lệ, giữ cân bằng).

## 3. Đặc trưng

- Log-mel spectrogram (n_mels=64, n_fft=1024, hop_length=256).
- MFCC + delta (mô tả lý do chọn).

## 4. Mô hình

- Kiến trúc CNN + BiLSTM (mô tả chi tiết, số tham số).
- Loss, optimizer, class weights.

## 5. Thực nghiệm

- Siêu tham số: epochs, batch_size, early stopping.
- Biểu đồ loss/accuracy; confusion matrix.
- Bảng kết quả: accuracy/precision/recall/F1 theo từng lớp.

## 6. Phân tích lỗi

- Mẫu bị nhầm lẫn, giả thuyết nguyên nhân (nội dung, cường độ, speaker).

## 7. Demo & Triển khai

- Cách chạy suy luận với file .wav mới.
- Gợi ý tích hợp thời gian thực (streaming).

## 8. Kết luận & Hướng phát triển

- Bài học rút ra.
- Cải tiến tương lai (SpecAugment, pretraining, Transformer, multi-task).

## Tài liệu tham khảo

- RAVDESS, EmoDB, Librosa, TensorFlow/Keras, và tài liệu khóa học.
