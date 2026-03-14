# Malaria-Prediction# Malaria Detection System - Deep Learning Based Diagnosis

A complete deep learning solution for automated malaria diagnosis from blood cell images using transfer learning.

## 📋 Project Overview

This system uses convolutional neural networks (CNN) with transfer learning to classify blood cell images as either:
- **Parasitized**: Cell infected with malaria parasite
- **Uninfected**: Healthy cell

The solution implements state-of-the-art deep learning techniques with data augmentation, transfer learning, and comprehensive evaluation metrics.

## 🏗️ Project Structure

```
malaria_detection/
├── requirements.txt              # Python dependencies
├── utils.py                      # Utility functions and helpers
├── data_preprocessing.py          # Data loading and preprocessing
├── model_training.py              # Model building and training
├── prediction.py                  # Prediction on new images
├── main.py                        # Main execution script
├── README.md                      # This file
├── data/                          # Data directory
│   ├── raw/                       # Raw images (Parasitized/, Uninfected/)
│   └── processed/                 # Preprocessed data arrays
├── models/                        # Trained models
│   ├── best_model.h5             # Best trained model
│   └── malaria_model.h5          # Final trained model
└── results/                       # Results and visualizations
    ├── training_history.png       # Training/validation curves
    ├── confusion_matrix.png       # Confusion matrix
    └── predictions_report.txt     # Prediction report
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your blood cell images in the following structure:

```
data/raw/
├── Parasitized/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Uninfected/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Run the Complete Pipeline

```bash
python main.py
```

This will:
- Load and preprocess all images
- Split data into train/validation/test sets
- Train the model using transfer learning
- Evaluate performance on test set
- Save trained model and results

### 4. Make Predictions on New Images

```python
from prediction import MalariaDiagnoser

# Initialize with trained model
diagnoser = MalariaDiagnoser('models/best_model.h5')

# Predict on single image
result = diagnoser.predict_single_image('path/to/image.jpg')
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict on directory of images
results = diagnoser.predict_from_directory('path/to/images/')

# Get report
report = diagnoser.generate_report(results, 'results/predictions_report.txt')
print(report)
```

## 📚 Module Documentation

### `utils.py`
Utility functions for image processing and visualization.

**Key Functions:**
- `load_image()`: Load and resize images
- `normalize_image()`: Normalize pixel values to [0, 1]
- `augment_image()`: Apply data augmentation
- `plot_training_history()`: Visualize training curves
- `plot_confusion_matrix()`: Visualize predictions
- `print_classification_metrics()`: Print performance metrics

### `data_preprocessing.py`
Data loading and preprocessing functionality.

**Key Class:** `MalariaDataPreprocessor`
- `load_images_from_directory()`: Load all images from organized folders
- `split_data()`: Split into train/validation/test (60%/20%/20%)
- `create_data_generators()`: Setup data augmentation
- `save_processed_data()`: Save arrays to disk
- `load_processed_data()`: Load previously saved data

**Example:**
```python
from data_preprocessing import MalariaDataPreprocessor

preprocessor = MalariaDataPreprocessor()
images, labels = preprocessor.load_images_from_directory('data/raw')
data_splits = preprocessor.split_data(images, labels)
```

### `model_training.py`
Model building, training, and evaluation.

**Key Class:** `MalariaDetectionModel`
- Supports multiple architectures: MobileNetV2, ResNet50, VGG16
- Transfer learning with frozen base weights
- Custom top layers for binary classification
- Early stopping and learning rate reduction callbacks

**Example:**
```python
from model_training import MalariaDetectionModel

model = MalariaDetectionModel(model_name='mobilenetv2')
model.build_model()
model.train(train_images, train_labels, val_images, val_labels, epochs=25)
model.evaluate(test_images, test_labels)
model.save_model()
```

### `prediction.py`
Making predictions on new blood cell images.

**Key Class:** `MalariaDiagnoser`
- Load trained models
- Predict on single images or batches
- Generate diagnostic reports
- Configurable prediction threshold (default 0.5)

**Example:**
```python
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')

# Single image
result = diagnoser.predict_single_image('cell.jpg')

# Batch of images
results = diagnoser.predict_batch(['cell1.jpg', 'cell2.jpg'])

# From directory
results = diagnoser.predict_from_directory('images/')

# Generate report
report = diagnoser.generate_report(results, save_path='report.txt')
```

## 🤖 Model Architecture

The system uses **Transfer Learning** with pre-trained weights:

```
Input (224x224x3)
    ↓
MobileNetV2 Base Model (frozen weights)
    ↓
Global Average Pooling
    ↓
Dense (256) + ReLU + Dropout(0.5)
    ↓
Dense (128) + ReLU + Dropout(0.3)
    ↓
Dense (1) + Sigmoid
    ↓
Output (Binary Classification)
```

**Architecture Benefits:**
- Leverages pre-trained ImageNet weights
- Reduces training time and data requirements
- Improves generalization
- MobileNetV2 is lightweight (efficient for deployment)

## 📊 Training Configuration

**Hyperparameters:**
- Optimizer: Adam (learning_rate=1e-4)
- Loss: Binary Crossentropy
- Batch Size: 32
- Epochs: 25 (with early stopping)
- Data Split: 60% Train / 20% Validation / 20% Test

**Data Augmentation:**
- Rotation: ±20°
- Width Shift: 20%
- Height Shift: 20%
- Zoom: 20%
- Horizontal/Vertical Flip

**Callbacks:**
- Early Stopping (patience=5)
- Model Checkpoint (best on validation accuracy)
- Learning Rate Reduction (factor=0.5, patience=3)

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall correctness
- **Precision**: True positives / All positives
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of predictions
- **ROC Curve**: Trade-off between TPR and FPR

## 🔍 Detailed Usage Examples

### Example 1: Complete Training from Scratch

```python
from data_preprocessing import preprocess_pipeline
from model_training import train_pipeline

# Preprocess data
data_splits = preprocess_pipeline('data/raw', save_processed=True)

# Train model
model, history = train_pipeline(
    train_images=data_splits['train_images'],
    train_labels=data_splits['train_labels'],
    val_images=data_splits['val_images'],
    val_labels=data_splits['val_labels'],
    test_images=data_splits['test_images'],
    test_labels=data_splits['test_labels'],
    model_name='mobilenetv2',
    epochs=30,
    batch_size=32
)
```

### Example 2: Batch Prediction with Report

```python
from prediction import MalariaDiagnoser
from pathlib import Path

diagnoser = MalariaDiagnoser('models/best_model.h5')

# Predict on all images in directory
results = diagnoser.predict_from_directory('new_images/')

# Generate formatted report
report = diagnoser.generate_report(results, 'results/batch_report.txt')

# Print summary
print(f"Total Images: {len(results)}")
parasitized = sum(1 for r in results if r['prediction'] == 'Parasitized')
print(f"Parasitized: {parasitized}")
print(f"Uninfected: {len(results) - parasitized}")
```

### Example 3: Custom Model with Different Architecture

```python
from model_training import MalariaDetectionModel

# Use ResNet50 instead of MobileNetV2
model = MalariaDetectionModel(model_name='resnet50')
model.build_model()

# Or VGG16
model = MalariaDetectionModel(model_name='vgg16')
model.build_model()

model.train(train_images, train_labels, val_images, val_labels, epochs=25)
```

### Example 4: Adjust Prediction Threshold

```python
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')

# Lower threshold = more sensitive to parasitized cells
diagnoser.set_threshold(0.4)

result = diagnoser.predict_single_image('cell.jpg')
```

## 🎯 Performance Expectations

Expected performance on well-balanced dataset:
- Accuracy: 85-95%
- Precision: 85-92%
- Recall: 87-95%
- F1-Score: 86-93%

*Actual performance depends on dataset quality, size, and image preprocessing*

## 📝 Output Files

After running the system:

1. **models/best_model.h5** - Best trained model (used for predictions)
2. **models/malaria_model.h5** - Final trained model
3. **results/training_history.png** - Accuracy and loss curves
4. **results/confusion_matrix.png** - Confusion matrix heatmap
5. **data/processed/** - Saved array files (train/val/test splits)
6. **predictions_report.txt** - Diagnostic report

## 🔧 Troubleshooting

**Issue: "Could not load image"**
- Ensure images are in supported formats (JPG, PNG, JPEG)
- Check file paths and permissions

**Issue: "Out of Memory"**
- Reduce batch_size (try 16 or 8)
- Reduce image_size to 128x128
- Use smaller model (MobileNetV2)

**Issue: "Poor accuracy"**
- Ensure sufficient training data (minimum 500+ images per class)
- Verify data is properly organized in directory structure
- Try different model architecture (ResNet50)
- Increase epochs or adjust learning rate

**Issue: "Model not found"**
- Verify model path is correct
- Ensure model was trained and saved

## 📦 Dependencies

- TensorFlow 2.13.0
- Keras 2.13.0
- NumPy 1.24.3
- OpenCV 4.8.0
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- Pillow 10.0.0

## 📌 Notes

- Models are trained with binary crossentropy loss for binary classification
- Images are normalized to [0, 1] range
- Transfer learning uses ImageNet pre-trained weights
- Predictions include confidence scores for reliability assessment
- System is optimized for CPU and GPU execution

## 🎓 Educational Resources

This project demonstrates:
- Transfer learning with pre-trained models
- Data augmentation techniques
- Model evaluation and validation
- Production-ready prediction pipeline
- Medical image classification

## 📄 License

This project is provided as-is for educational and research purposes.

## 👨‍💼 Support

For issues, questions, or improvements, please refer to the code documentation or modify as needed.

---

**Last Updated:** March 2026
**Version:** 1.0
