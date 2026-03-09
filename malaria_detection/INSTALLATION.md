# Installation & Setup Guide

## Quick Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA 11.0+ and cuDNN for GPU support

### Step 1: Install Dependencies

Navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

This installs all required packages:
- TensorFlow/Keras (deep learning)
- OpenCV (image processing)
- NumPy/Pandas (data handling)
- Scikit-learn (ML metrics)
- Matplotlib (visualization)

### Step 2: Verify Installation

Test the installation:

```bash
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### Step 3: Organize Your Data

Create the following directory structure:

```
malaria_detection/
└── data/
    └── raw/
        ├── Parasitized/      (put infected cell images here)
        └── Uninfected/       (put healthy cell images here)
```

Example:
```
data/raw/
├── Parasitized/
│   ├── cell_001.jpg
│   ├── cell_002.jpg
│   └── ...
└── Uninfected/
    ├── cell_001.jpg
    ├── cell_002.jpg
    └── ...
```

### Step 4: Run the System

```bash
python main.py
```

This will:
1. Load images from data/raw/
2. Preprocess and split data
3. Train the model
4. Evaluate on test set
5. Save results to models/ and results/ directories

## Detailed Setup Instructions

### For Windows Users

1. **Install Python**
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv malaria_env
   malaria_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run System**
   ```bash
   python main.py
   ```

### For macOS Users

1. **Install Python**
   ```bash
   brew install python3
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv malaria_env
   source malaria_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run System**
   ```bash
   python main.py
   ```

### For Linux Users

1. **Install Python**
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip python3-venv
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv malaria_env
   source malaria_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run System**
   ```bash
   python main.py
   ```

## GPU Support (Optional)

For faster training on NVIDIA GPU:

### Install CUDA
1. Download CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Follow installation instructions for your OS
3. Add CUDA to PATH

### Install cuDNN
1. Download cuDNN: https://developer.nvidia.com/cudnn
2. Extract and add to CUDA directory

### Verify GPU Setup
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Update TensorFlow for GPU
```bash
pip install --upgrade tensorflow[and-cuda]
```

## Troubleshooting Installation

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
pip install --upgrade pip
pip install tensorflow==2.13.0
```

### Issue: "ImportError: DLL load failed"

**Solution (Windows):**
- Download Visual C++ Redistributable: https://support.microsoft.com/en-us/help/2977003
- Reinstall TensorFlow: `pip install --force-reinstall tensorflow==2.13.0`

### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python==4.8.0.74 --force-reinstall
```

### Issue: "CUDA not found" (on GPU systems)

**Solution:**
1. Install CUDA and cuDNN properly
2. Add to environment variables:
   - CUDA_PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
3. Restart Python/IDE

### Issue: "Out of Memory" during training

**Solution:**
1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 16  # or 8
   ```

2. Reduce image size:
   ```python
   IMAGE_SIZE = (128, 128)
   ```

3. Use lighter model:
   ```python
   MODEL_NAME = 'mobilenetv2'
   ```

### Issue: "CUDA out of memory" (on GPU)

**Solution:**
```bash
# Reduce GPU memory usage
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"
```

## Testing Installation

After installation, test the complete pipeline:

```bash
# Test data preprocessing
python -c "from data_preprocessing import MalariaDataPreprocessor; print('✓ Data preprocessing module OK')"

# Test model training
python -c "from model_training import MalariaDetectionModel; print('✓ Model training module OK')"

# Test predictions
python -c "from prediction import MalariaDiagnoser; print('✓ Prediction module OK')"

# Test utilities
python -c "import utils; print('✓ Utils module OK')"
```

## Verify Complete Setup

```bash
python -c "
import sys
import importlib

modules = [
    'tensorflow', 'keras', 'numpy', 'cv2', 
    'pandas', 'sklearn', 'matplotlib', 'PIL'
]

print('Checking installed packages:')
for module in modules:
    try:
        mod = importlib.import_module(module)
        print(f'✓ {module}')
    except ImportError:
        print(f'✗ {module} NOT FOUND')

print('\nChecking project modules:')
try:
    import utils
    print('✓ utils.py')
except ImportError as e:
    print(f'✗ utils.py: {e}')

try:
    import data_preprocessing
    print('✓ data_preprocessing.py')
except ImportError as e:
    print(f'✗ data_preprocessing.py: {e}')

try:
    import model_training
    print('✓ model_training.py')
except ImportError as e:
    print(f'✗ model_training.py: {e}')

try:
    import prediction
    print('✓ prediction.py')
except ImportError as e:
    print(f'✗ prediction.py: {e}')
"
```

## Next Steps

1. **Prepare Data**: Organize blood cell images in `data/raw/`
2. **Run Training**: Execute `python main.py`
3. **Make Predictions**: Check `QUICKSTART.md` for usage examples
4. **Explore**: Review `examples.py` for advanced usage

## System Requirements

### Minimum Requirements
- CPU: Intel i5 or equivalent
- RAM: 8GB
- Disk: 2GB free space
- Python 3.8+

### Recommended Requirements
- CPU: Intel i7 or equivalent
- RAM: 16GB+
- GPU: NVIDIA GTX 1060 or better
- Disk: 10GB free space
- Python 3.9+

## Environment Variables (Optional)

Create a `.env` file for custom settings:

```bash
# GPU Settings
CUDA_VISIBLE_DEVICES=0
TF_CPP_MIN_LOG_LEVEL=2

# Data Paths
DATA_RAW_PATH=data/raw
DATA_PROCESSED_PATH=data/processed

# Model
MODEL_PATH=models/best_model.h5
```

Then load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv('DATA_RAW_PATH', 'data/raw')
```

## Uninstallation

To remove and clean up:

```bash
# Deactivate virtual environment
deactivate  # or 'malaria_env\Scripts\deactivate' on Windows

# Remove virtual environment
rm -rf malaria_env  # or 'rmdir /s malaria_env' on Windows

# Uninstall packages
pip uninstall -r requirements.txt -y
```

## Support & Help

For issues:
1. Check `README.md` for detailed documentation
2. Review `QUICKSTART.md` for common tasks
3. See `examples.py` for usage patterns
4. Check error messages for specific solutions

---

**Installation Guide Version**: 1.0  
**Last Updated**: March 2026  
**Compatible With**: Python 3.8, 3.9, 3.10, 3.11
