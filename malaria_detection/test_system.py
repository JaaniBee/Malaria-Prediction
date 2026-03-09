"""
Simple test script to verify malaria detection system is working
Run this to test basic functionality without long training
"""

import sys
print("Starting basic functionality test...\n")

# Test 1: Import all modules
print("=" * 70)
print("TEST 1: Checking all imports...")
print("=" * 70)

try:
    print("[OK] Importing TensorFlow...", end=" ")
    import tensorflow as tf
    print("OK")
    
    print("[OK] Importing NumPy...", end=" ")
    import numpy as np
    print("OK")
    
    print("[OK] Importing OpenCV...", end=" ")
    import cv2
    print("OK")
    
    print("[OK] Importing Pandas...", end=" ")
    import pandas as pd
    print("OK")
    
    print("[OK] Importing Scikit-learn...", end=" ")
    from sklearn.metrics import accuracy_score
    print("OK")
    
    print("[OK] Importing Matplotlib...", end=" ")
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("OK")
    
    print("[OK] Importing Seaborn...", end=" ")
    import seaborn as sns
    print("OK")
    
    print("[OK] All core packages loaded successfully!\n")
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    sys.exit(1)

# Test 2: Import project modules
print("=" * 70)
print("TEST 2: Checking project modules...")
print("=" * 70)

try:
    print("[OK] Importing utils...", end=" ")
    import utils
    print("OK")
    
    print("[OK] Importing data_preprocessing...", end=" ")
    import data_preprocessing
    print("OK")
    
    print("[OK] Importing model_training...", end=" ")
    import model_training
    print("OK")
    
    print("[OK] Importing prediction...", end=" ")
    import prediction
    print("OK")
    
    print("[OK] Importing config...", end=" ")
    import config
    print("OK")
    
    print("[OK] All project modules loaded successfully!\n")
    
except ImportError as e:
    print(f"\n[ERROR] Import Error: {e}")
    sys.exit(1)

# Test 3: Create directory structure
print("=" * 70)
print("TEST 3: Creating directory structure...")
print("=" * 70)

try:
    from pathlib import Path
    
    dirs = ['data', 'models', 'results', 'predictions']
    for d in dirs:
        path = Path(d)
        path.mkdir(exist_ok=True)
        print(f"[OK] Created/verified {d}/ folder")
    
    print("[OK] Directory structure ready!\n")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)

# Test 4: Create demo data
print("=" * 70)
print("TEST 4: Creating demo data for testing...")
print("=" * 70)

try:
    from pathlib import Path
    
    # Create dummy images for testing
    parasitized_dir = Path("data/raw/Parasitized")
    uninfected_dir = Path("data/raw/Uninfected")
    
    parasitized_dir.mkdir(parents=True, exist_ok=True)
    uninfected_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 dummy images for each class
    for i in range(10):
        # Parasitized cell (random image)
        img_para = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        cv2.imwrite(f"data/raw/Parasitized/cell_{i:03d}.jpg", 
                   cv2.cvtColor(img_para, cv2.COLOR_RGB2BGR))
        
        # Uninfected cell (random image)
        img_unin = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        cv2.imwrite(f"data/raw/Uninfected/cell_{i:03d}.jpg", 
                   cv2.cvtColor(img_unin, cv2.COLOR_RGB2BGR))
    
    print(f"[OK] Created 10 dummy Parasitized images")
    print(f"[OK] Created 10 dummy Uninfected images")
    print("[OK] Demo data ready!\n")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)

# Test 5: Test data preprocessing
print("=" * 70)
print("TEST 5: Testing data preprocessing...")
print("=" * 70)

try:
    from data_preprocessing import MalariaDataPreprocessor
    
    preprocessor = MalariaDataPreprocessor(image_size=(224, 224))
    print("[OK] Initializing preprocessor...")
    
    images, labels = preprocessor.load_images_from_directory('data/raw')
    print(f"[OK] Loaded {len(images)} images")
    print(f"  - Shape: {images.shape}")
    print(f"  - Parasitized: {np.sum(labels==0)}")
    print(f"  - Uninfected: {np.sum(labels==1)}")
    
    data_splits = preprocessor.split_data(images, labels)
    print(f"[OK] Data split successfully")
    print(f"  - Train: {len(data_splits['train_images'])}")
    print(f"  - Val: {len(data_splits['val_images'])}")
    print(f"  - Test: {len(data_splits['test_images'])}")
    print("[OK] Data preprocessing works!\n")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Quick model test
print("=" * 70)
print("TEST 6: Testing model building...")
print("=" * 70)

try:
    from model_training import MalariaDetectionModel
    
    model = MalariaDetectionModel(model_name='mobilenetv2')
    print("[OK] Initializing model...")
    
    model.build_model()
    print("[OK] Model built successfully")
    print(f"  - Architecture: MobileNetV2")
    print(f"  - Parameters: {model.model.count_params():,}")
    print("[OK] Model building works!\n")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour malaria detection system is ready to use!")
print("\nNext steps:")
print("1. Run: python main.py")
print("2. Check results/ folder for outputs")
print("3. Use prediction.py to diagnose images")
print("\n" + "=" * 70)
