"""
Quick Start Guide - 5 minutes to malaria detection
Minimal example to get you started
"""

# ==============================================================================
# OPTION 1: RUN EVERYTHING IN ONE GO
# ==============================================================================
"""
Running this will:
1. Load and preprocess data from 'data/raw/'
2. Train the model
3. Evaluate on test set
4. Save model and results

Command: python main.py
"""


# ==============================================================================
# OPTION 2: STEP-BY-STEP IN PYTHON
# ==============================================================================

# Step 1: Preprocess Data
from data_preprocessing import preprocess_pipeline

data_splits = preprocess_pipeline(
    data_directory='data/raw',
    save_processed=True,
    save_path='data/processed'
)


# Step 2: Train Model
from model_training import train_pipeline

model, history = train_pipeline(
    train_images=data_splits['train_images'],
    train_labels=data_splits['train_labels'],
    val_images=data_splits['val_images'],
    val_labels=data_splits['val_labels'],
    test_images=data_splits['test_images'],
    test_labels=data_splits['test_labels'],
    model_name='mobilenetv2',  # Light and fast
    epochs=25,
    batch_size=32
)


# Step 3: Make Predictions
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')

# Single image
result = diagnoser.predict_single_image('path/to/image.jpg')
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Multiple images
results = diagnoser.predict_from_directory('path/to/images/')
report = diagnoser.generate_report(results, 'results/report.txt')


# ==============================================================================
# OPTION 3: MINIMAL EXAMPLE
# ==============================================================================

# Just make predictions on new images
from prediction import MalariaDiagnoser
import numpy as np

diagnoser = MalariaDiagnoser('models/best_model.h5')

# Predict on sample images
results = diagnoser.predict_from_directory('test_images/')

for result in results:
    print(f"{result['image_path']}: {result['prediction']} ({result['confidence']:.1%})")


# ==============================================================================
# DATA ORGANIZATION
# ==============================================================================
"""
Before running, organize your data like this:

data/
└── raw/
    ├── Parasitized/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    └── Uninfected/
        ├── image_1.jpg
        ├── image_2.jpg
        └── ...

The system will automatically:
- Load all images
- Normalize pixel values
- Split into train/val/test
- Apply data augmentation
- Train the model
- Save results
"""


# ==============================================================================
# QUICK CONFIGURATION
# ==============================================================================

# Change settings in config.py before running:
from config import get_config

# Quick training (5 epochs, fast)
quick_config = get_config('quick')

# Full training (50 epochs, best results)
full_config = get_config('full')

# Light model (mobile-friendly)
light_config = get_config('light')


# ==============================================================================
# COMMON TASKS
# ==============================================================================

# 1. PREDICT AND GET REPORT
from prediction import MalariaDiagnoser

diagnoser = MalariaDiagnoser('models/best_model.h5')
results = diagnoser.predict_from_directory('images/')
report = diagnoser.generate_report(results, 'report.txt')


# 2. BATCH PROCESSING
import os
from pathlib import Path

image_dir = Path('images/')
for image_file in image_dir.glob('*.jpg'):
    result = diagnoser.predict_single_image(str(image_file))
    print(f"{image_file.name}: {result['prediction']}")


# 3. ADJUST SENSITIVITY
diagnoser.set_threshold(0.4)  # More sensitive to parasitized
# or
diagnoser.set_threshold(0.6)  # Less sensitive to parasitized


# 4. VIEW RESULTS
import pandas as pd

# Convert results to DataFrame for analysis
results_list = []
for result in results:
    results_list.append({
        'image': result['image_path'],
        'diagnosis': result['prediction'],
        'confidence': result['confidence'],
        'score': result['raw_score']
    })

df = pd.DataFrame(results_list)
print(df.to_string())

# Summary statistics
parasitized_count = (df['diagnosis'] == 'Parasitized').sum()
uninfected_count = (df['diagnosis'] == 'Uninfected').sum()
avg_confidence = df['confidence'].mean()

print(f"\nSummary:")
print(f"Parasitized: {parasitized_count}")
print(f"Uninfected: {uninfected_count}")
print(f"Average Confidence: {avg_confidence:.2%}")


# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

"""
ISSUE: "No module named 'tensorflow'"
SOLUTION: pip install -r requirements.txt

ISSUE: "Could not find data"
SOLUTION: Ensure data is in data/raw/Parasitized/ and data/raw/Uninfected/

ISSUE: "Model not found"
SOLUTION: Train model first or check model path

ISSUE: "Out of memory"
SOLUTION: Reduce batch_size in config.py (try 16 or 8)

ISSUE: "Slow performance"
SOLUTION: 
- Use MobileNetV2 (already default)
- Reduce image_size to 128x128
- Reduce epochs
- Use GPU if available
"""


# ==============================================================================
# NEXT STEPS
# ==============================================================================

"""
1. Read README.md for detailed documentation
2. Examine examples.py for more use cases
3. Check model_training.py for model customization
4. Look at config.py for all configurable options
5. Explore utils.py for additional utilities
"""


# ==============================================================================
# FILE STRUCTURE
# ==============================================================================

"""
malaria_detection/
├── main.py                      # Run everything
├── data_preprocessing.py         # Data handling
├── model_training.py             # Model training
├── prediction.py                # Make predictions
├── utils.py                     # Helper functions
├── config.py                    # Configuration
├── examples.py                  # Usage examples
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
│
├── data/
│   ├── raw/                    # Your images here
│   │   ├── Parasitized/
│   │   └── Uninfected/
│   └── processed/              # Auto-generated
│
├── models/
│   ├── best_model.h5           # Auto-saved
│   └── malaria_model.h5        # Auto-saved
│
└── results/
    ├── training_history.png    # Auto-generated
    ├── confusion_matrix.png    # Auto-generated
    └── predictions_report.txt  # Auto-generated
"""


# ==============================================================================
# EXAMPLE OUTPUT
# ==============================================================================

"""
RUNNING main.py produces:

==============================================================================
MALARIA DETECTION SYSTEM - DEEP LEARNING BASED DIAGNOSIS
==============================================================================

Step 1: Setting up project structure...
✓ Directory structure ready

Step 2: Data preprocessing...
Loading 500 images from Parasitized...
Loading 500 images from Uninfected...
✓ Loaded 1000 images

✓ Data split completed

Training set: 600 images
Validation set: 200 images
Testing set: 200 images

✓ Processed data saved

Step 3: Model training...
Building mobilenetv2 model...
✓ Model built and compiled successfully!
Total parameters: 2,259,329

Training...
Epoch 1/25
19/19 [==============================] - 65s 3s/step - loss: 0.5432 - accuracy: 0.7450 - val_loss: 0.3221 - val_accuracy: 0.8650
...
Epoch 25/25
19/19 [==============================] - 31s 2s/step - loss: 0.1234 - accuracy: 0.9650 - val_loss: 0.1876 - val_accuracy: 0.9350

✓ Model training completed

Step 4: Making predictions...
Test Set Metrics:
  Accuracy:  0.9350
  Precision: 0.9234
  Recall:    0.9456
  F1-Score:  0.9344

EXECUTION SUMMARY
==============================================================================
✓ Preprocessed 1000 images
✓ Trained model with 2,259,329 parameters
✓ Achieved 93.50% accuracy on test set
✓ Model saved to: models/best_model.h5
✓ Results saved to: results/
"""


# ==============================================================================
# GETTING HELP
# ==============================================================================

"""
For detailed help on any module:

from data_preprocessing import MalariaDataPreprocessor
help(MalariaDataPreprocessor)

from model_training import MalariaDetectionModel
help(MalariaDetectionModel)

from prediction import MalariaDiagnoser
help(MalariaDiagnoser)

Check docstrings in each file for function documentation.
"""


if __name__ == "__main__":
    print("QUICK START GUIDE - Malaria Detection System")
    print("=" * 70)
    print("\n1. Prepare your data in data/raw/Parasitized/ and data/raw/Uninfected/")
    print("2. Run: python main.py")
    print("3. Use prediction.py to make diagnoses on new images")
    print("\nFor more details, see README.md and examples.py")
    print("=" * 70)
