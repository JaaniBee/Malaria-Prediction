"""
Example usage script for malaria detection system
Demonstrates common workflows and use cases
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_preprocessing import MalariaDataPreprocessor
from model_training import MalariaDetectionModel
from prediction import MalariaDiagnoser
import utils


def example_1_load_and_preprocess():
    """
    Example 1: Load and preprocess images
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: LOAD AND PREPROCESS IMAGES")
    print("="*70 + "\n")
    
    preprocessor = MalariaDataPreprocessor(image_size=(224, 224))
    
    # Load images from directory structure
    # Expected: data/raw/Parasitized/*.jpg and data/raw/Uninfected/*.jpg
    try:
        images, labels = preprocessor.load_images_from_directory('data/raw')
        print(f"[OK] Loaded {len(images)} images")
        print(f"  Shape: {images.shape}")
        print(f"  Classes: Parasitized ({np.sum(labels==0)}), Uninfected ({np.sum(labels==1)})")
        
        # Split data
        splits = preprocessor.split_data(images, labels)
        print(f"[OK] Data split completed")
        print(f"  Training: {len(splits['train_images'])} images")
        print(f"  Validation: {len(splits['val_images'])} images")
        print(f"  Testing: {len(splits['test_images'])} images")
        
        return splits
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Note: Ensure data is in proper directory structure")
        return None


def example_2_build_model():
    """
    Example 2: Build different model architectures
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: BUILD MODEL ARCHITECTURES")
    print("="*70 + "\n")
    
    # List available models
    models = ['mobilenetv2', 'resnet50', 'vgg16']
    
    for model_name in models:
        print(f"\nBuilding {model_name.upper()} model...")
        try:
            model = MalariaDetectionModel(model_name=model_name)
            model.build_model()
            print(f"[OK] {model_name} built successfully")
            print(f"  Total parameters: {model.model.count_params():,}")
        except Exception as e:
            print(f"[ERROR] Error: {e}")


def example_3_train_model(data_splits):
    """
    Example 3: Train the model
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: TRAIN MODEL")
    print("="*70 + "\n")
    
    if data_splits is None:
        print("Skipping - no data available")
        return None
    
    try:
        # Build model
        model = MalariaDetectionModel(model_name='mobilenetv2')
        model.build_model()
        
        # Train
        history = model.train(
            train_images=data_splits['train_images'],
            train_labels=data_splits['train_labels'],
            val_images=data_splits['val_images'],
            val_labels=data_splits['val_labels'],
            epochs=10,  # Reduced for example
            batch_size=32
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(data_splits['test_images'], data_splits['test_labels'])
        print(f"\n[OK] Training completed")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Loss: {loss:.4f}")
        
        # Save model
        model.save_model('models/example_model.h5')
        print(f"[OK] Model saved to models/example_model.h5")
        
        return model
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None


def example_4_predict_single_image():
    """
    Example 4: Make prediction on single image
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: PREDICT ON SINGLE IMAGE")
    print("="*70 + "\n")
    
    try:
        diagnoser = MalariaDiagnoser('models/best_model.h5')
        
        # Create dummy image for demonstration
        dummy_image_path = 'test_image.jpg'
        
        # Create a sample image if not exists
        if not Path(dummy_image_path).exists():
            print("Creating sample image for demonstration...")
            from PIL import Image
            img_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            Image.fromarray(img_array).save(dummy_image_path)
        
        result = diagnoser.predict_single_image(dummy_image_path)
        
        print(f"[OK] Prediction completed")
        print(f"  Image: {result['image_path']}")
        print(f"  Diagnosis: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Raw Score: {result['raw_score']:.4f}")
        
        # Cleanup
        if Path(dummy_image_path).exists():
            Path(dummy_image_path).unlink()
        
        return result
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("  Ensure model exists at models/best_model.h5")
        return None


def example_5_batch_prediction():
    """
    Example 5: Make predictions on batch of images
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: BATCH PREDICTION WITH REPORT")
    print("="*70 + "\n")
    
    try:
        diagnoser = MalariaDiagnoser('models/best_model.h5')
        
        # Create dummy images
        print("Creating sample images...")
        dummy_dir = Path('dummy_images')
        dummy_dir.mkdir(exist_ok=True)
        
        from PIL import Image
        image_paths = []
        for i in range(5):
            img_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            img_path = dummy_dir / f'sample_{i}.jpg'
            Image.fromarray(img_array).save(img_path)
            image_paths.append(str(img_path))
        
        # Make predictions
        results = diagnoser.predict_batch(image_paths)
        
        # Generate report
        report = diagnoser.generate_report(results, 'results/example_report.txt')
        print(report)
        
        # Cleanup
        import shutil
        shutil.rmtree(dummy_dir)
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")


def example_6_visualize_results():
    """
    Example 6: Visualize results and metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: VISUALIZE RESULTS")
    print("="*70 + "\n")
    
    print("Creating sample visualizations...")
    
    try:
        # Create dummy prediction data
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0])
        
        # Plot confusion matrix
        utils.plot_confusion_matrix(
            y_true, y_pred,
            class_names=['Parasitized', 'Uninfected'],
            save_path='results/example_confusion_matrix.png'
        )
        print("[OK] Confusion matrix saved")
        
        # Print classification metrics
        utils.print_classification_metrics(
            y_true, y_pred,
            class_names=['Parasitized', 'Uninfected']
        )
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")


def example_7_adjust_threshold():
    """
    Example 7: Adjust prediction threshold
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: ADJUST PREDICTION THRESHOLD")
    print("="*70 + "\n")
    
    try:
        diagnoser = MalariaDiagnoser('models/best_model.h5')
        
        print("Default threshold (0.5):")
        print(f"  Current: {diagnoser.threshold}")
        
        # Set higher threshold (less sensitive to parasitized)
        print("\nSetting threshold to 0.6...")
        diagnoser.set_threshold(0.6)
        print(f"  Current: {diagnoser.threshold}")
        
        # Set lower threshold (more sensitive to parasitized)
        print("\nSetting threshold to 0.4...")
        diagnoser.set_threshold(0.4)
        print(f"  Current: {diagnoser.threshold}")
        
        # Reset to default
        diagnoser.set_threshold(0.5)
        print(f"\nReset to default: {diagnoser.threshold}")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")


def example_8_custom_data_augmentation():
    """
    Example 8: Use custom data augmentation
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: DATA AUGMENTATION")
    print("="*70 + "\n")
    
    # Create sample image
    sample_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    sample_image_normalized = utils.normalize_image(sample_image)
    
    print("Applying augmentation transformations...")
    
    try:
        # Apply augmentation
        augmented = utils.augment_image(
            sample_image_normalized,
            rotation_range=20,
            shift_range=0.2,
            zoom_range=0.2
        )
        
        print(f"[OK] Original shape: {sample_image.shape}")
        print(f"[OK] Augmented shape: {augmented.shape}")
        print("[OK] Augmentation applied successfully")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(sample_image_normalized)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(augmented)
        axes[1].set_title('Augmented')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/augmentation_example.png', dpi=100, bbox_inches='tight')
        print("[OK] Visualization saved to results/augmentation_example.png")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("MALARIA DETECTION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    # Example 1: Load and preprocess
    data_splits = example_1_load_and_preprocess()
    
    # Example 2: Build models
    example_2_build_model()
    
    # Example 3: Train model (if data available)
    if data_splits:
        model = example_3_train_model(data_splits)
    
    # Example 4: Single prediction
    example_4_predict_single_image()
    
    # Example 5: Batch prediction
    example_5_batch_prediction()
    
    # Example 6: Visualizations
    example_6_visualize_results()
    
    # Example 7: Threshold adjustment
    example_7_adjust_threshold()
    
    # Example 8: Data augmentation
    example_8_custom_data_augmentation()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
