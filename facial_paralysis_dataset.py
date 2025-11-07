import random
import torch
import torchvision
import torch.utils.data as data
import cv2
import numpy as np
import os
import json
from torch.utils.data import Dataset
import torch.nn.functional as F


class FacialParalysisDataset(Dataset):
    """
    Dataset for facial paralysis detection and grading
    Supports both binary classification (normal/paralysis) and multi-class grading
    """
    
    def __init__(self, data_dir, phase='train', task_type='classification', 
                 num_classes=2, res=128, augment=True):
        """
        Args:
            data_dir: Directory containing facial paralysis images and labels
            phase: 'train' or 'test'
            task_type: 'classification' or 'grading'
            num_classes: Number of classes (2 for binary, >2 for grading)
            res: Input resolution
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.phase = phase
        self.task_type = task_type
        self.num_classes = num_classes
        self.res = res
        self.augment = augment
        
        # Load data and labels
        self.samples = self._load_samples()
        
        # Transform
        self.transform = torchvision.transforms.ToTensor()
        
        print(f"Loaded {len(self.samples)} samples for {phase}")
        
    def _load_samples(self):
        """Load image paths and labels from dataset directory"""
        samples = []
        
        # Expected directory structure:
        # data_dir/
        #   normal/
        #   paralysis/
        #     mild/
        #     moderate/
        #     severe/
        # labels.json (optional, for custom label mapping)
        
        # Try to load from structured directories
        if self.task_type == 'classification' and self.num_classes == 2:
            # Binary classification: normal vs paralysis
            normal_dir = os.path.join(self.data_dir, 'normal')
            paralysis_dir = os.path.join(self.data_dir, 'paralysis')
            
            if os.path.exists(normal_dir) and os.path.exists(paralysis_dir):
                # Load normal samples (label 0)
                for img_file in os.listdir(normal_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        samples.append({
                            'image_path': os.path.join(normal_dir, img_file),
                            'label': 0
                        })
                
                # Load paralysis samples (label 1)
                for img_file in os.listdir(paralysis_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        samples.append({
                            'image_path': os.path.join(paralysis_dir, img_file),
                            'label': 1
                        })
        
        elif self.task_type == 'grading' and self.num_classes > 2:
            # Multi-class grading
            grade_dirs = ['normal', 'mild', 'moderate', 'severe'][:self.num_classes]
            
            for grade_idx, grade_dir in enumerate(grade_dirs):
                full_path = os.path.join(self.data_dir, grade_dir)
                if os.path.exists(full_path):
                    for img_file in os.listdir(full_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            samples.append({
                                'image_path': os.path.join(full_path, img_file),
                                'label': grade_idx
                            })
        
        # If no structured directories found, try to load from labels.json
        if not samples:
            labels_file = os.path.join(self.data_dir, 'labels.json')
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)
                
                for item in labels_data:
                    img_path = os.path.join(self.data_dir, item['image'])
                    if os.path.exists(img_path):
                        samples.append({
                            'image_path': img_path,
                            'label': item['label']
                        })
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")
            
        return samples
    
    def _preprocess_image(self, image):
        """Preprocess image: face detection, alignment, and normalization"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Simple face detection and cropping (you may want to use a more sophisticated method)
        # For now, we'll resize directly to the target resolution
        h, w = image.shape[:2]
        
        # Center crop to square
        if h != w:
            if h > w:
                start = (h - w) // 2
                image = image[start:start+w, :]
            else:
                start = (w - h) // 2
                image = image[:, start:start+h]
        
        # Resize to target resolution
        image = cv2.resize(image, (self.res, self.res))
        
        return image
    
    def _augment_image(self, image):
        """Apply data augmentation for training"""
        if self.phase == 'train' and self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image[:, ::-1]
            
            # Random brightness and contrast adjustment
            if random.random() > 0.5:
                # Brightness
                brightness_factor = random.uniform(0.8, 1.2)
                image = np.clip(image * brightness_factor, 0, 255)
            
            if random.random() > 0.5:
                # Contrast
                contrast_factor = random.uniform(0.8, 1.2)
                mean = np.mean(image)
                image = np.clip((image - mean) * contrast_factor + mean, 0, 255)
            
            # Random rotation
            if random.random() > 0.7:
                angle = random.uniform(-15, 15)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {sample['image_path']}")
        
        # Preprocess
        image = self._preprocess_image(image)
        
        # Augment
        image = self._augment_image(image)
        
        # Convert to tensor
        image = self.transform(image)
        
        # Get label
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, label