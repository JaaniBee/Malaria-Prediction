"""
Utility functions for malaria detection system
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns


def load_image(image_path, target_size=(224, 224)):
    """
    Load an image from file and resize it to target size.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (height, width)
    
    Returns:
        numpy.ndarray: Loaded and resized image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    return image


def normalize_image(image):
    """
    Normalize image to values between 0 and 1.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Normalized image
    """
    return image.astype('float32') / 255.0


def augment_image(image, rotation_range=20, shift_range=0.2, zoom_range=0.2):
    """
    Apply data augmentation to an image.
    
    Args:
        image (numpy.ndarray): Input image
        rotation_range (int): Rotation range in degrees
        shift_range (float): Shift range as fraction of total
        zoom_range (float): Zoom range as fraction
    
    Returns:
        numpy.ndarray: Augmented image
    """
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h))
    
    # Random shift
    dx = int(w * np.random.uniform(-shift_range, shift_range))
    dy = int(h * np.random.uniform(-shift_range, shift_range))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, matrix, (w, h))
    
    return image


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history object from model training
        save_path (str): Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[OK] Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Names of classes
        save_path (str): Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_metrics(y_true, y_pred, class_names):
    """
    Print classification metrics including precision, recall, F1-score.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Names of classes
    """
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))


def create_directory_structure(base_path):
    """
    Create directory structure for the project.
    
    Args:
        base_path (str): Base path for the project
    """
    directories = ['data', 'models', 'results', 'predictions']
    for directory in directories:
        (Path(base_path) / directory).mkdir(parents=True, exist_ok=True)
