"""
Prediction module for malaria detection
Makes predictions on new blood cell images
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import cv2
from tensorflow.keras.models import load_model  # type: ignore
import utils


class MalariaDiagnoser:
    """Handles making predictions on new images"""
    
    def __init__(self, model_path='models/best_model.h5'):
        """
        Initialize the diagnoser with a trained model.
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model = load_model(model_path)
        self.class_names = ['Parasitized', 'Uninfected']
        self.threshold = 0.5
        
    def predict_single_image(self, image_path):
        """
        Make prediction on a single image.
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        image = utils.load_image(image_path, target_size=(224, 224))
        image = utils.normalize_image(image)
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)[0][0]
        
        # Determine class
        if prediction > self.threshold:
            predicted_class = 'Uninfected'
            confidence = prediction
        else:
            predicted_class = 'Parasitized'
            confidence = 1 - prediction
        
        return {
            'image_path': str(image_path),
            'prediction': predicted_class,
            'confidence': float(confidence),
            'raw_score': float(prediction)
        }
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images.
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        return results
    
    def predict_from_directory(self, directory_path, file_extensions=['jpg', 'jpeg', 'png']):
        """
        Make predictions on all images in a directory.
        
        Args:
            directory_path (str): Path to directory with images
            file_extensions (list): Image file extensions to consider
        
        Returns:
            list: List of prediction results
        """
        directory_path = Path(directory_path)
        image_paths = []
        
        for ext in file_extensions:
            image_paths.extend(directory_path.glob(f'*.{ext}'))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.predict_batch(image_paths)
    
    def predict_array(self, image_array):
        """
        Make predictions on a numpy array of images.
        
        Args:
            image_array (numpy.ndarray): Array of images (N, H, W, C)
        
        Returns:
            list: List of predictions
        """
        predictions_raw = self.model.predict(image_array, verbose=0)
        
        results = []
        for i, pred in enumerate(predictions_raw):
            score = pred[0]
            if score > self.threshold:
                predicted_class = 'Uninfected'
                confidence = score
            else:
                predicted_class = 'Parasitized'
                confidence = 1 - score
            
            results.append({
                'image_index': i,
                'prediction': predicted_class,
                'confidence': float(confidence),
                'raw_score': float(score)
            })
        
        return results
    
    def generate_report(self, results, save_path=None):
        """
        Generate a text report of predictions.
        
        Args:
            results (list): List of prediction results
            save_path (str): Optional path to save report
        
        Returns:
            str: Formatted report
        """
        report = "="*70 + "\n"
        report += "MALARIA DIAGNOSIS REPORT\n"
        report += "="*70 + "\n\n"
        
        parasitized_count = 0
        uninfected_count = 0
        total_confidence_parasitized = 0
        total_confidence_uninfected = 0
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                report += f"Image {i}: {result['image_path']}\n"
                report += f"  ERROR: {result['error']}\n\n"
                continue
            
            report += f"Image {i}: {Path(result['image_path']).name}\n"
            report += f"  Diagnosis: {result['prediction']}\n"
            report += f"  Confidence: {result['confidence']:.2%}\n"
            report += f"  Raw Score: {result['raw_score']:.4f}\n\n"
            
            if result['prediction'] == 'Parasitized':
                parasitized_count += 1
                total_confidence_parasitized += result['confidence']
            else:
                uninfected_count += 1
                total_confidence_uninfected += result['confidence']
        
        # Summary
        report += "="*70 + "\n"
        report += "SUMMARY\n"
        report += "="*70 + "\n"
        report += f"Total Images: {len(results)}\n"
        report += f"Parasitized: {parasitized_count}\n"
        report += f"Uninfected: {uninfected_count}\n"
        
        if parasitized_count > 0:
            report += f"Avg Confidence (Parasitized): {total_confidence_parasitized/parasitized_count:.2%}\n"
        if uninfected_count > 0:
            report += f"Avg Confidence (Uninfected): {total_confidence_uninfected/uninfected_count:.2%}\n"
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def set_threshold(self, threshold):
        """
        Set prediction threshold (default 0.5).
        
        Args:
            threshold (float): New threshold value (0-1)
        """
        if 0 <= threshold <= 1:
            self.threshold = threshold
            print(f"Threshold set to {threshold}")
        else:
            print("Threshold must be between 0 and 1")


def predict_pipeline(model_path, image_source, report_path=None):
    """
    Complete prediction pipeline.
    
    Args:
        model_path (str): Path to trained model
        image_source (str): Path to image or directory
        report_path (str): Optional path to save report
    
    Returns:
        list: Prediction results
    """
    print("="*50)
    print("LOADING MODEL")
    print("="*50)
    diagnoser = MalariaDiagnoser(model_path)
    
    source_path = Path(image_source)
    
    if source_path.is_dir():
        print("\n" + "="*50)
        print("PREDICTING ON DIRECTORY")
        print("="*50)
        results = diagnoser.predict_from_directory(image_source)
    else:
        print("\n" + "="*50)
        print("PREDICTING ON SINGLE IMAGE")
        print("="*50)
        results = [diagnoser.predict_single_image(image_source)]
    
    # Generate report
    report = diagnoser.generate_report(results, report_path)
    print("\n" + report)
    
    return results


if __name__ == "__main__":
    print("Prediction module loaded successfully!")
