# Lab 4 – Speech Emotion Recognition (SER)

## Cách dùng nhanh (Colab)

1. Tải notebook `Lab4_Speech_Emotion_Recognition.ipynb` lên Colab.
2. Chuẩn bị dữ liệu:
   - RAVDESS: `datasets/ravdess/...`
   - EmoDB: `datasets/emodb/...`
   - Cập nhật đường dẫn trong biến `DATA_DIRS` ở notebook.
3. Chạy tuần tự các cell.
4. Kết quả:
   - Model: `ser_cnn_bilstm.h5`
   - Biểu đồ training và confusion matrix trong output.
   - Hàm `predict_wav(path)` để test file .wav của bạn.

## Ghi chú

- Nên dùng GPU.
- Nếu lớp quá ít mẫu (<20), notebook tự loại khỏi huấn luyện để tránh overfit.
- Muốn tái tạo đúng rubric: bổ sung demo video ngắn ghi lại inference và đính kèm báo cáo.
