"""
Data preprocessing module for malaria detection
Handles loading, preprocessing, and augmentation of blood cell images
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import utils


class MalariaDataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, image_size=(224, 224), test_split=0.2, val_split=0.2):
        """
        Initialize the preprocessor.
        
        Args:
            image_size (tuple): Target size for all images
            test_split (float): Fraction of data for testing
            val_split (float): Fraction of data for validation
        """
        self.image_size = image_size
        self.test_split = test_split
        self.val_split = val_split
        self.class_names = ['Parasitized', 'Uninfected']
        
    def load_images_from_directory(self, directory_path):
        """
        Load all images from a directory structure.
        Expected structure: directory_path/class_name/*.jpg
        
        Args:
            directory_path (str): Path to directory containing class folders
        
        Returns:
            tuple: (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        directory_path = Path(directory_path)
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = directory_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class directory not found: {class_path}")
                continue
            
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for image_path in image_files:
                try:
                    image = utils.load_image(image_path, self.image_size)
                    image = utils.normalize_image(image)
                    images.append(image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def split_data(self, images, labels):
        """
        Split data into training, validation, and testing sets.
        
        Args:
            images (numpy.ndarray): All images
            labels (numpy.ndarray): All labels
        
        Returns:
            dict: Dictionary containing train, val, and test splits with their labels
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=self.test_split, random_state=42, stratify=labels
        )
        
        # Second split: separate validation set from training
        val_split_adjusted = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split summary:")
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Testing set: {len(X_test)} images")
        
        return {
            'train_images': X_train,
            'train_labels': y_train,
            'val_images': X_val,
            'val_labels': y_val,
            'test_images': X_test,
            'test_labels': y_test
        }
    
    def create_data_generators(self):
        """
        Create data augmentation generators for training and validation.
        
        Returns:
            tuple: (train_generator, val_generator)
        """
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def save_processed_data(self, data_dict, save_path):
        """
        Save processed data to disk for later use.
        
        Args:
            data_dict (dict): Dictionary containing data splits
            save_path (str): Path to save the data
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in data_dict.items():
            np.save(save_path / f"{key}.npy", value)
        
        print(f"Data saved to {save_path}")
    
    def load_processed_data(self, load_path):
        """
        Load previously processed data from disk.
        
        Args:
            load_path (str): Path to load the data from
        
        Returns:
            dict: Dictionary containing data splits
        """
        load_path = Path(load_path)
        data_dict = {}
        
        keys = ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']
        for key in keys:
            data_dict[key] = np.load(load_path / f"{key}.npy")
        
        print(f"Data loaded from {load_path}")
        return data_dict


def preprocess_pipeline(data_directory, save_processed=True, save_path='data/processed'):
    """
    Complete preprocessing pipeline.
    
    Args:
        data_directory (str): Path to raw data directory
        save_processed (bool): Whether to save processed data
        save_path (str): Path to save processed data
    
    Returns:
        dict: Dictionary containing all data splits
    """
    preprocessor = MalariaDataPreprocessor()
    
    print("="*50)
    print("LOADING DATA")
    print("="*50)
    images, labels = preprocessor.load_images_from_directory(data_directory)
    
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    data_splits = preprocessor.split_data(images, labels)
    
    if save_processed:
        preprocessor.save_processed_data(data_splits, save_path)
    
    return data_splits


if __name__ == "__main__":
    # Example usage
    # data_splits = preprocess_pipeline('data/raw')
    print("Data preprocessing module loaded successfully!")
