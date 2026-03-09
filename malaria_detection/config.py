"""
Configuration file for malaria detection system
Easily modify these settings for different scenarios and experiments
"""

# ==================== DATA CONFIGURATION ====================

# Path to raw data directory
# Expected structure: data/raw/Parasitized/*.jpg and data/raw/Uninfected/*.jpg
DATA_RAW_PATH = 'data/raw'

# Path to save processed data
DATA_PROCESSED_PATH = 'data/processed'

# Image size for all models
IMAGE_SIZE = (224, 224)

# Data split ratios
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2


# ==================== MODEL CONFIGURATION ====================

# Base model architecture
# Options: 'mobilenetv2' (recommended, lightweight), 'resnet50', 'vgg16'
MODEL_NAME = 'mobilenetv2'

# Freeze base model weights (transfer learning)
FREEZE_BASE_WEIGHTS = True

# Number of neurons in custom dense layers
DENSE_LAYER_1 = 256
DENSE_LAYER_2 = 128

# Dropout rates
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3


# ==================== TRAINING CONFIGURATION ====================

# Number of training epochs
EPOCHS = 25

# Batch size for training
BATCH_SIZE = 32

# Learning rate for Adam optimizer
LEARNING_RATE = 1e-4

# Early stopping patience (stop if no improvement for N epochs)
EARLY_STOPPING_PATIENCE = 5

# Learning rate reduction factor
LR_REDUCTION_FACTOR = 0.5

# Learning rate reduction patience
LR_REDUCTION_PATIENCE = 3

# Minimum learning rate
MIN_LEARNING_RATE = 1e-7


# ==================== DATA AUGMENTATION CONFIGURATION ====================

# Rotation range in degrees
AUGMENTATION_ROTATION_RANGE = 20

# Width shift range (fraction of total width)
AUGMENTATION_WIDTH_SHIFT = 0.2

# Height shift range (fraction of total height)
AUGMENTATION_HEIGHT_SHIFT = 0.2

# Zoom range (fraction)
AUGMENTATION_ZOOM_RANGE = 0.2

# Enable horizontal flip
AUGMENTATION_HORIZONTAL_FLIP = True

# Enable vertical flip
AUGMENTATION_VERTICAL_FLIP = True


# ==================== PREDICTION CONFIGURATION ====================

# Path to trained model for predictions
MODEL_SAVE_PATH = 'models/best_model.h5'

# Prediction threshold (values > threshold = Uninfected, <= threshold = Parasitized)
PREDICTION_THRESHOLD = 0.5

# Path for prediction results
RESULTS_PATH = 'results'


# ==================== OUTPUT PATHS ====================

# Directory for saving models
MODELS_PATH = 'models'

# Directory for saving results (plots, reports, etc.)
RESULTS_PATH = 'results'

# Directory for saving data
DATA_PATH = 'data'


# ==================== DEVICE CONFIGURATION ====================

# Use GPU if available (True/False)
USE_GPU = True

# Number of CPU threads (0 = auto)
NUM_CPU_THREADS = 0


# ==================== LOGGING CONFIGURATION ====================

# Verbose output level (0=silent, 1=progress bar, 2=one line per epoch)
VERBOSE_LEVEL = 1

# Save training logs
SAVE_LOGS = True

# Log file path
LOG_FILE = 'results/training.log'


# ==================== CLASS LABELS ====================

# Class names (must match directory names)
CLASS_NAMES = ['Parasitized', 'Uninfected']

# Class indices
CLASS_INDICES = {
    'Parasitized': 0,
    'Uninfected': 1
}


# ==================== METRICS CONFIGURATION ====================

# Metrics to track during training
METRICS = ['accuracy']

# Loss function
LOSS_FUNCTION = 'binary_crossentropy'

# Optimizer
OPTIMIZER = 'adam'


# ==================== EXPERIMENTAL CONFIGURATIONS ====================

# Quick training configuration (for testing)
QUICK_TRAIN_CONFIG = {
    'epochs': 5,
    'batch_size': 16,
    'freeze_base': True,
    'model_name': 'mobilenetv2'
}

# Full training configuration (for production)
FULL_TRAIN_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'freeze_base': True,
    'model_name': 'resnet50',
    'unfreeze_after_epochs': 30  # Unfreeze base after N epochs
}

# Light model configuration (mobile/edge devices)
LIGHT_MODEL_CONFIG = {
    'model_name': 'mobilenetv2',
    'image_size': (128, 128),
    'batch_size': 16,
    'dense_1': 128,
    'dense_2': 64
}


# ==================== UTILITY FUNCTIONS ====================

def get_config(config_type='default'):
    """
    Get predefined configuration set.
    
    Args:
        config_type (str): 'default', 'quick', 'full', or 'light'
    
    Returns:
        dict: Configuration dictionary
    """
    if config_type == 'quick':
        return {
            'epochs': QUICK_TRAIN_CONFIG['epochs'],
            'batch_size': QUICK_TRAIN_CONFIG['batch_size'],
            'model_name': QUICK_TRAIN_CONFIG['model_name'],
        }
    elif config_type == 'full':
        return {
            'epochs': FULL_TRAIN_CONFIG['epochs'],
            'batch_size': FULL_TRAIN_CONFIG['batch_size'],
            'model_name': FULL_TRAIN_CONFIG['model_name'],
        }
    elif config_type == 'light':
        return {
            'image_size': LIGHT_MODEL_CONFIG['image_size'],
            'batch_size': LIGHT_MODEL_CONFIG['batch_size'],
            'model_name': LIGHT_MODEL_CONFIG['model_name'],
        }
    else:  # 'default'
        return {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'model_name': MODEL_NAME,
            'image_size': IMAGE_SIZE,
            'learning_rate': LEARNING_RATE,
        }


def print_config():
    """Print current configuration"""
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"\nDATA:")
    print(f"  Raw path: {DATA_RAW_PATH}")
    print(f"  Processed path: {DATA_PROCESSED_PATH}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Train/Val/Test split: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    
    print(f"\nMODEL:")
    print(f"  Architecture: {MODEL_NAME}")
    print(f"  Freeze base: {FREEZE_BASE_WEIGHTS}")
    print(f"  Dense layers: {DENSE_LAYER_1}, {DENSE_LAYER_2}")
    
    print(f"\nTRAINING:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    print(f"\nAUGMENTATION:")
    print(f"  Rotation: ±{AUGMENTATION_ROTATION_RANGE}°")
    print(f"  Shift: {AUGMENTATION_WIDTH_SHIFT*100}%")
    print(f"  Zoom: {AUGMENTATION_ZOOM_RANGE*100}%")
    
    print(f"\nPREDICTION:")
    print(f"  Model path: {MODEL_SAVE_PATH}")
    print(f"  Threshold: {PREDICTION_THRESHOLD}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
    print("\nExample usage:")
    print("  from config import get_config")
    print("  config = get_config('quick')  # Get quick training config")
    print("  config = get_config('full')   # Get full training config")
