"""
Model training module for malaria detection using deep learning
Uses transfer learning with pre-trained models for optimal performance
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import utils


class MalariaDetectionModel:
    """Handles model creation and training"""
    
    def __init__(self, model_name='mobilenetv2', input_shape=(224, 224, 3)):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of base model ('mobilenetv2', 'resnet50', 'vgg16')
            input_shape (tuple): Input image shape
        """
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, freeze_base=True):
        """
        Build the model using transfer learning.
        
        Args:
            freeze_base (bool): Whether to freeze base model weights
        
        Returns:
            tensorflow.keras.models.Model: Compiled model
        """
        print(f"Building {self.model_name} model...")
        
        # Load pre-trained base model
        if self.model_name == 'mobilenetv2':
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'resnet50':
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'vgg16':
            base_model = VGG16(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze base model weights
        if freeze_base:
            base_model.trainable = False
            print("Base model weights frozen")
        
        # Add custom top layers
        inputs = base_model.input
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built and compiled successfully!")
        print(f"Total parameters: {self.model.count_params()}")
        
        return self.model
    
    def train(self, train_images, train_labels, val_images, val_labels, 
              epochs=25, batch_size=32, callbacks=None):
        """
        Train the model.
        
        Args:
            train_images (numpy.ndarray): Training images
            train_labels (numpy.ndarray): Training labels
            val_images (numpy.ndarray): Validation images
            val_labels (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of callbacks
        
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        if self.model is None:
            self.build_model()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(train_images, train_labels, batch_size=batch_size),
            steps_per_epoch=len(train_images) // batch_size,
            epochs=epochs,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_images, test_labels):
        """
        Evaluate the model on test set.
        
        Args:
            test_images (numpy.ndarray): Test images
            test_labels (numpy.ndarray): Test labels
        
        Returns:
            tuple: (loss, accuracy)
        """
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        loss, accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict(self, images):
        """
        Make predictions on new images.
        
        Args:
            images (numpy.ndarray): Images to predict on
        
        Returns:
            numpy.ndarray: Predictions (probabilities)
        """
        predictions = self.model.predict(images, verbose=0)
        return predictions
    
    def save_model(self, save_path='models/malaria_model.h5'):
        """
        Save the model to disk.
        
        Args:
            save_path (str): Path to save the model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """
        Load a previously trained model from disk.
        
        Args:
            load_path (str): Path to load the model from
        """
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
        self.model = keras_load_model(load_path)
        print(f"Model loaded from {load_path}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            print("Model not built yet!")
            return
        self.model.summary()


def train_pipeline(train_images, train_labels, val_images, val_labels, 
                   test_images, test_labels, model_name='mobilenetv2', 
                   epochs=25, batch_size=32):
    """
    Complete training pipeline.
    
    Args:
        train_images, train_labels: Training data
        val_images, val_labels: Validation data
        test_images, test_labels: Test data
        model_name (str): Name of model to use
        epochs (int): Number of epochs
        batch_size (int): Batch size
    
    Returns:
        tuple: (model, history)
    """
    # Build and train model
    model = MalariaDetectionModel(model_name=model_name)
    model.build_model()
    
    history = model.train(
        train_images, train_labels,
        val_images, val_labels,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    model.evaluate(test_images, test_labels)
    
    # Plot training history
    utils.plot_training_history(history, save_path='results/training_history.png')
    
    # Save model
    model.save_model()
    
    return model, history


if __name__ == "__main__":
    print("Model training module loaded successfully!")
