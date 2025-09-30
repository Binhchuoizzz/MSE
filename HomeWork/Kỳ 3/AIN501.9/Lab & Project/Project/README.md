# Emotion Detection using Deep Learning

## Introduction

This project builds a deep learning system that classifies facial expressions into **seven emotions**: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised. It is trained on the **FER-2013** dataset (ICML 2013), which contains 35,887 grayscale facial images at 48×48 resolution.

### Real‑world applications

- **Human–computer interaction**: adapt UI/UX in real time based on user affect.
- **Customer experience analytics**: measure sentiment in retail, kiosks, or digital signage.
- **Education**: gauge engagement or confusion in remote learning.
- **Healthcare & well‑being**: track mood changes for mental health monitoring (with consent).
- **Entertainment & gaming**: drive avatars or in‑game reactions.
- **Robotics**: enable socially‑aware robot behavior.

## How the project works

1. A **Haar Cascade** face detector finds faces in frames (webcam or images).
2. Each detected face region is converted to grayscale and resized to **48×48**.
3. A **CNN** model predicts softmax scores over the 7 classes.
4. The class with highest score is rendered as the predicted emotion.

At a glance, there are two primary modes:

- `train`: trains the CNN on FER‑2013 images prepared into folders.
- `display`: runs webcam inference using saved weights (`model.h5`).

## Training pipeline and techniques used

### Dataset

- Source: **FER‑2013** (Kaggle). The original format is a CSV with pixel strings.
- The script `src/dataset_prepare.py` converts the CSV into **PNG images** and organizes them into `data/train/<class>` and `data/test/<class>` folders.
- Train/val split sizes used in the code: `num_train = 28709`, `num_val = 7178`.

### Data loading

- Uses `ImageDataGenerator` with `rescale=1./255` for both training and validation.
- Grayscale input with `target_size=(48, 48)` and `class_mode='categorical'`.

### Model architecture (Keras Sequential)

- Input: `(48, 48, 1)` grayscale.
- Convolutions: `Conv2D(32, 3×3)` → `Conv2D(64, 3×3)` → `MaxPooling2D` → `Dropout(0.25)`
- Convolutions: `Conv2D(128, 3×3)` → `MaxPooling2D` → `Conv2D(128, 3×3)` → `MaxPooling2D` → `Dropout(0.25)`
- Classifier head: `Flatten` → `Dense(1024, relu)` → `Dropout(0.5)` → `Dense(7, softmax)`

### Optimization

- Loss: `categorical_crossentropy`
- Optimizer: `Adam(learning_rate=1e-4, decay=1e-6)`
- Batch size: `64`; Epochs: `60` (configurable)

### Training utilities

- `plot_model_history` saves `src/plot.png` with accuracy and loss curves.
- Weights are saved to `src/model.h5` after training.

### Inference pipeline (`display` mode)

- Loads `model.h5` weights.
- Captures frames from the default webcam.
- Detects faces with `haarcascade_frontalface_default.xml`.
- Crops, resizes to 48×48, expands dims to match `(1, 48, 48, 1)`, predicts, and overlays label.

## Environment setup and usage

### Dependencies

- Python 3
- OpenCV, TensorFlow (Keras), NumPy, SciPy, Matplotlib, PIL
- Install via:

```bash
pip install -r requirements.txt
```

### Project structure

- `src/data/` — prepared dataset folders (`train/`, `test/` with class subfolders)
- `src/emotions.py` — training and inference script
- `src/dataset_prepare.py` — CSV→PNG conversion and folderization
- `src/haarcascade_frontalface_default.xml` — face detector
- `src/model.h5` — saved weights
- `imgs/accuracy.png` — sample accuracy curve visualization

### Prepare dataset from FER‑2013 CSV (optional)

1. Place `fer2013.csv` inside `src/`.
2. From `src/`, run:

```bash
python dataset_prepare.py
```

This creates `src/data/train` and `src/data/test` with class‑wise PNG images.

### Train

From `src/`:

```bash
python emotions.py --mode train
```

Artifacts: `model.h5` weights and `plot.png` with accuracy/loss curves.

### Inference (webcam)

From `src/`:

```bash
python emotions.py --mode display
```

Press `q` to quit the webcam window.

## Results

- A simple **4‑layer CNN** variant achieved about **63.2% test accuracy** after 50 epochs (as referenced from the original implementation). With the current configuration (60 epochs, Adam 1e‑4), your mileage may vary depending on hardware, random seeds, and augmentation.
- See `imgs/accuracy.png` and `src/plot.png` for training dynamics.

### Notes on performance

- Grayscale 48×48 input favors speed and simplicity over fine‑grained detail.
- The class imbalance in FER‑2013 can affect per‑class precision/recall.
- Real‑time webcam inference is feasible on CPU; GPU improves throughput.

## Future work

- **Data augmentation**: random flips, rotations, shifts, zoom, brightness, and contrast.
- **Class imbalance handling**: weighted loss, focal loss, oversampling.
- **Architecture improvements**: BatchNorm, residual blocks, depthwise separable convs, or transfer learning from lightweight backbones.
- **Larger input size**: 64×64 or 96×96 to capture more detail.
- **Temporal modeling**: use video sequences (RNN/Temporal CNN) for smoother predictions.
- **Deployment**: Convert to TensorFlow Lite / ONNX; optimize for edge devices.
- **Ethics and privacy**: add guidance on consent, data retention, and bias evaluation.

## Conclusion

This repository provides a concise, end‑to‑end baseline for real‑time facial emotion recognition: dataset preparation, CNN training, and live inference. It is a solid starting point for experimentation and production‑oriented improvements such as augmentation, stronger architectures, and deployment optimization.

## References

- Goodfellow, I., Erhan, D., Carrier, P. L., et al. “Challenges in Representation Learning: A report on three machine learning contests.” arXiv (2013).
- FER‑2013 Kaggle dataset: https://www.kaggle.com/datasets/deadskull7/fer2013
