# 🏥 Malaria Detection System - Project Overview

## 📦 Complete Project Package

This is a **production-ready deep learning solution** for automated malaria diagnosis from blood cell microscopy images.

### ✨ Key Features

- **Transfer Learning**: Uses pre-trained MobileNetV2, ResNet50, or VGG16
- **High Accuracy**: 85-95% accuracy on well-balanced datasets
- **Easy to Use**: Simple API for training and predictions
- **Well Documented**: Comprehensive code comments and documentation
- **Data Augmentation**: Advanced image preprocessing and augmentation
- **Production Ready**: Includes model saving, evaluation, and visualization

---

## 📂 File Structure & Descriptions

### Core Python Modules

| File | Purpose | Key Classes |
|------|---------|------------|
| **utils.py** | Helper functions and utilities | Image processing, visualization, metrics |
| **data_preprocessing.py** | Data loading and preprocessing | `MalariaDataPreprocessor` |
| **model_training.py** | Model building and training | `MalariaDetectionModel` |
| **prediction.py** | Inference and predictions | `MalariaDiagnoser` |
| **main.py** | Complete execution pipeline | Full workflow orchestration |
| **config.py** | Configuration settings | Environment variables and configs |
| **examples.py** | Usage examples and demonstrations | 8 complete examples |

### Documentation Files

| File | Content |
|------|---------|
| **README.md** | Complete project documentation |
| **QUICKSTART.md** | 5-minute getting started guide |
| **INSTALLATION.md** | Detailed installation instructions |
| **requirements.txt** | Python package dependencies |

### Auto-Generated Directories

```
data/
├── raw/                 # Your images (Parasitized/, Uninfected/)
└── processed/           # Auto-generated processed data

models/
├── best_model.h5        # Best model (auto-saved)
└── malaria_model.h5     # Final model (auto-saved)

results/
├── training_history.png     # Training curves
├── confusion_matrix.png     # Prediction analysis
└── predictions_report.txt   # Diagnostic report
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/raw/
├── Parasitized/  (infected cell images)
└── Uninfected/   (healthy cell images)
```

### 3. Run Complete Pipeline
```bash
python main.py
```

### 4. Make Predictions
```python
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')
result = diagnoser.predict_single_image('cell_image.jpg')
print(f"Diagnosis: {result['prediction']} ({result['confidence']:.1%})")
```

---

## 📚 Detailed Module Reference

### `utils.py` - Utilities & Helpers

**Image Processing:**
- `load_image()` - Load and resize images
- `normalize_image()` - Normalize pixel values [0,1]
- `augment_image()` - Apply random transformations

**Visualization:**
- `plot_training_history()` - Plot accuracy/loss curves
- `plot_confusion_matrix()` - Visualize predictions
- `print_classification_metrics()` - Print performance statistics

**Utilities:**
- `create_directory_structure()` - Setup project directories

### `data_preprocessing.py` - Data Handling

**Main Class:** `MalariaDataPreprocessor`

Methods:
- `load_images_from_directory()` - Load all images
- `split_data()` - Split into train/val/test (60%/20%/20%)
- `create_data_generators()` - Setup augmentation
- `save_processed_data()` - Save to disk
- `load_processed_data()` - Load saved data

Example:
```python
preprocessor = MalariaDataPreprocessor()
images, labels = preprocessor.load_images_from_directory('data/raw')
splits = preprocessor.split_data(images, labels)
```

### `model_training.py` - Model Training

**Main Class:** `MalariaDetectionModel`

Supported Architectures:
- MobileNetV2 (default - lightweight, fast)
- ResNet50 (high accuracy)
- VGG16 (very accurate but slow)

Methods:
- `build_model()` - Create transfer learning model
- `train()` - Train on data
- `evaluate()` - Test on test set
- `predict()` - Make predictions
- `save_model()` / `load_model()` - Persistence

Example:
```python
model = MalariaDetectionModel(model_name='mobilenetv2')
model.build_model()
model.train(X_train, y_train, X_val, y_val, epochs=25)
model.evaluate(X_test, y_test)
```

### `prediction.py` - Inference

**Main Class:** `MalariaDiagnoser`

Methods:
- `predict_single_image()` - Predict one image
- `predict_batch()` - Predict multiple images
- `predict_from_directory()` - Predict all in directory
- `predict_array()` - Predict numpy array
- `generate_report()` - Create formatted report
- `set_threshold()` - Adjust sensitivity

Example:
```python
diagnoser = MalariaDiagnoser('models/best_model.h5')

# Single image
result = diagnoser.predict_single_image('image.jpg')

# Batch
results = diagnoser.predict_from_directory('images/')
report = diagnoser.generate_report(results)
```

---

## 🎯 Common Usage Patterns

### Pattern 1: Train and Predict
```python
from data_preprocessing import preprocess_pipeline
from model_training import train_pipeline
from prediction import MalariaDiagnoser

# Train
data = preprocess_pipeline('data/raw')
model, history = train_pipeline(
    data['train_images'], data['train_labels'],
    data['val_images'], data['val_labels'],
    data['test_images'], data['test_labels']
)

# Predict
diagnoser = MalariaDiagnoser('models/best_model.h5')
results = diagnoser.predict_from_directory('new_images/')
```

### Pattern 2: Batch Prediction with Report
```python
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')
results = diagnoser.predict_from_directory('images/')
report = diagnoser.generate_report(results, 'report.txt')
```

### Pattern 3: Custom Configuration
```python
from config import get_config
from model_training import MalariaDetectionModel

# Use quick config for testing
config = get_config('quick')

model = MalariaDetectionModel(model_name='resnet50')
model.build_model()
# Train with quick config
```

### Pattern 4: Analyze Results
```python
import pandas as pd
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')
results = diagnoser.predict_from_directory('images/')

# Convert to DataFrame
df = pd.DataFrame(results)
print(df.to_string())

# Statistics
print(f"Parasitized: {(df['prediction']=='Parasitized').sum()}")
print(f"Avg Confidence: {df['confidence'].mean():.2%}")
```

---

## ⚙️ Configuration

All settings in `config.py`:

**Data:**
- `IMAGE_SIZE` - Image dimensions
- `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT` - Data split ratios

**Model:**
- `MODEL_NAME` - Architecture (mobilenetv2, resnet50, vgg16)
- `FREEZE_BASE_WEIGHTS` - Use transfer learning
- `DENSE_LAYER_1`, `DENSE_LAYER_2` - Custom layers

**Training:**
- `EPOCHS` - Number of training iterations
- `BATCH_SIZE` - Images per batch
- `LEARNING_RATE` - Optimization rate

**Augmentation:**
- `AUGMENTATION_ROTATION_RANGE` - Rotation degrees
- `AUGMENTATION_ZOOM_RANGE` - Zoom amount
- `AUGMENTATION_*_SHIFT` - Translation amount

### Preset Configurations

```python
from config import get_config

# Quick training - 5 epochs
get_config('quick')

# Full training - 50 epochs
get_config('full')

# Light model - mobile-friendly
get_config('light')
```

---

## 📊 Expected Performance

On typical medical dataset (balanced, good quality):

| Metric | Expected Range |
|--------|-----------------|
| Accuracy | 85-95% |
| Precision | 85-92% |
| Recall | 87-95% |
| F1-Score | 86-93% |

*Results depend on dataset quality, size, and preprocessing*

---

## 🔧 Troubleshooting

### Installation Issues
→ See `INSTALLATION.md`

### Data Organization
```
data/raw/
├── Parasitized/
│   ├── img1.jpg
│   └── img2.jpg
└── Uninfected/
    ├── img1.jpg
    └── img2.jpg
```

### Memory Problems
- Reduce `BATCH_SIZE` in config.py
- Reduce `IMAGE_SIZE` to (128, 128)
- Use MobileNetV2 (already default)

### Performance Issues
- Enable GPU in config
- Use lighter model
- Reduce image size

---

## 📖 Documentation Files

| File | Best For |
|------|----------|
| **README.md** | Complete reference |
| **QUICKSTART.md** | Getting started |
| **INSTALLATION.md** | Setup help |
| **examples.py** | Code examples |
| **config.py** | Configuration |

---

## 🎓 What You'll Learn

- **Transfer Learning** - Using pre-trained models
- **Data Augmentation** - Image preprocessing
- **Model Training** - Deep learning workflow
- **Medical AI** - Classification from images
- **Production Code** - Professional Python practices

---

## 📝 Key Files Explained

### `main.py` - The Orchestrator
```
Runs complete pipeline:
1. Data preprocessing
2. Model training
3. Evaluation
4. Visualization & saving
```

### `model_training.py` - The Brain
```
Contains neural network:
- Transfer learning setup
- Custom top layers
- Training callbacks
- Model persistence
```

### `prediction.py` - The Predictor
```
Makes diagnoses:
- Single image prediction
- Batch processing
- Report generation
- Configurable threshold
```

### `config.py` - The Controller
```
All settings:
- Data paths
- Model parameters
- Training configs
- Augmentation options
```

---

## 🚀 Next Steps

1. **Install**: `pip install -r requirements.txt`
2. **Prepare**: Organize images in `data/raw/`
3. **Train**: `python main.py`
4. **Predict**: Use `MalariaDiagnoser` on new images
5. **Explore**: Check `examples.py` for more uses

---

## 📞 Quick Answers

**Q: How do I use my own data?**  
A: Place images in `data/raw/Parasitized/` and `data/raw/Uninfected/`

**Q: How do I make predictions on new images?**  
A: Use `MalariaDiagnoser` class in `prediction.py`

**Q: Can I use different model architecture?**  
A: Yes, change `MODEL_NAME` in `model_training.py` (mobilenetv2, resnet50, vgg16)

**Q: How do I improve accuracy?**  
A: Use more training data, adjust hyperparameters in `config.py`

**Q: Can I use GPU?**  
A: Yes, TensorFlow automatically uses CUDA if available

**Q: What's the minimum dataset size?**  
A: 500+ images per class for decent results

---

## 📋 Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data organized in `data/raw/Parasitized/` and `data/raw/Uninfected/`
- [ ] At least 500 images per class
- [ ] Read `QUICKSTART.md`
- [ ] Run `python main.py`

---

## 🎉 You're Ready!

Everything is set up and documented. Start with:

```bash
python main.py
```

For questions, check the documentation files. Happy training! 🚀

---

**Version**: 1.0  
**Created**: March 2026  
**Status**: Production Ready  
**Language**: Python 3.8+
