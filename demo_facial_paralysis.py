import torch
import cv2
import numpy as np
import argparse
import os
from model import FAN
from Config import get_CTranS_config
import torchvision.transforms as transforms
from PIL import Image

class FacialParalysisClassifier:
    def __init__(self, model_path, num_classes=2, device='cuda'):
        """
        Initialize the facial paralysis classifier
        
        Args:
            model_path: Path to the trained model checkpoint
            num_classes: Number of classes (2 for binary, >2 for grading)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.config = get_CTranS_config()
        
        # Initialize model
        self.model = FAN(self.config, 3, 81, num_classes=num_classes, 
                        task_type='classification').to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Define class labels
        if num_classes == 2:
            self.class_labels = ['Normal', 'Facial Paralysis']
        elif num_classes == 4:
            self.class_labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        else:
            self.class_labels = [f'Class {i}' for i in range(num_classes)]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess input image for inference
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume it's a numpy array
            image = Image.fromarray(image_path).convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict(self, image_path, return_probabilities=False):
        """
        Predict facial paralysis class for an input image
        
        Args:
            image_path: Path to the input image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted class and optional probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            outputs, _ = self.model(image_tensor)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            
        if return_probabilities:
            return self.class_labels[predicted_class], probabilities.cpu().numpy()[0]
        else:
            return self.class_labels[predicted_class]
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict facial paralysis class for multiple images
        
        Args:
            image_paths: List of paths to input images
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of predicted classes and optional probabilities
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(None)
        
        return results
    
    def evaluate_directory(self, directory_path, return_probabilities=False):
        """
        Evaluate all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with image paths as keys and predictions as values
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(directory_path):
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_files.append(os.path.join(directory_path, file))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return {}
        
        print(f"Found {len(image_files)} images to evaluate...")
        
        # Predict for each image
        results = {}
        for image_file in image_files:
            try:
                prediction = self.predict(image_file, return_probabilities)
                results[image_file] = prediction
                print(f"{os.path.basename(image_file)}: {prediction}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results[image_file] = None
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Facial Paralysis Classification Demo')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory path')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (2 for binary, >2 for grading)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--return_probs', action='store_true',
                        help='Return class probabilities')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = FacialParalysisClassifier(
        model_path=args.model_path,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Check if input is directory or single file
    if os.path.isdir(args.input):
        # Evaluate directory
        results = classifier.evaluate_directory(args.input, args.return_probs)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                for image_path, prediction in results.items():
                    f.write(f"{image_path}: {prediction}\n")
            print(f"Results saved to {args.output}")
            
    else:
        # Single image prediction
        if os.path.exists(args.input):
            result = classifier.predict(args.input, args.return_probs)
            print(f"Input: {args.input}")
            print(f"Prediction: {result}")
            
            # Save result if output file specified
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(f"{args.input}: {result}\n")
                print(f"Result saved to {args.output}")
        else:
            print(f"Input file not found: {args.input}")


if __name__ == '__main__':
    main()