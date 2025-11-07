import os
import json
import shutil
import random
from pathlib import Path

class FacialParalysisDataPreparator:
    """
    Utility class to prepare facial paralysis dataset for training and testing
    """
    
    def __init__(self, root_dir):
        """
        Initialize the data preparator
        
        Args:
            root_dir: Root directory containing the facial paralysis data
        """
        self.root_dir = Path(root_dir)
        
    def organize_binary_classification(self, normal_dir, paralysis_dir, 
                                     train_ratio=0.8, seed=42):
        """
        Organize data for binary classification (Normal vs Facial Paralysis)
        
        Expected structure:
        root_dir/
        ├── normal/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── paralysis/
            ├── img1.jpg
            ├── img2.jpg
            └── ...
            
        Output structure:
        root_dir/
        ├── train/
        │   ├── normal/
        │   └── paralysis/
        └── test/
            ├── normal/
            └── paralysis/
        """
        random.seed(seed)
        
        # Create output directories
        train_dir = self.root_dir / 'train'
        test_dir = self.root_dir / 'test'
        
        for split_dir in [train_dir, test_dir]:
            for class_dir in ['normal', 'paralysis']:
                (split_dir / class_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for class_name, source_dir in [('normal', normal_dir), ('paralysis', paralysis_dir)]:
            source_path = Path(source_dir)
            if not source_path.exists():
                print(f"Warning: Source directory {source_path} does not exist")
                continue
                
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(source_path.glob(ext))
                image_files.extend(source_path.glob(ext.upper()))
            
            if not image_files:
                print(f"Warning: No images found in {source_path}")
                continue
            
            # Shuffle and split
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Copy files
            for file_path in train_files:
                shutil.copy2(file_path, train_dir / class_name / file_path.name)
            
            for file_path in test_files:
                shutil.copy2(file_path, test_dir / class_name / file_path.name)
            
            print(f"Class '{class_name}': {len(train_files)} train, {len(test_files)} test images")
    
    def organize_multi_class_grading(self, class_dirs, class_names=None,
                                   train_ratio=0.8, seed=42):
        """
        Organize data for multi-class grading (Normal, Mild, Moderate, Severe)
        
        Args:
            class_dirs: List of directories containing images for each class
            class_names: List of class names (optional, defaults to ['normal', 'mild', 'moderate', 'severe'])
        """
        if class_names is None:
            class_names = ['normal', 'mild', 'moderate', 'severe']
        
        if len(class_dirs) != len(class_names):
            raise ValueError("Number of class directories must match number of class names")
        
        random.seed(seed)
        
        # Create output directories
        train_dir = self.root_dir / 'train'
        test_dir = self.root_dir / 'test'
        
        for split_dir in [train_dir, test_dir]:
            for class_name in class_names:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for class_name, source_dir in zip(class_names, class_dirs):
            source_path = Path(source_dir)
            if not source_path.exists():
                print(f"Warning: Source directory {source_path} does not exist")
                continue
                
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(source_path.glob(ext))
                image_files.extend(source_path.glob(ext.upper()))
            
            if not image_files:
                print(f"Warning: No images found in {source_path}")
                continue
            
            # Shuffle and split
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Copy files
            for file_path in train_files:
                shutil.copy2(file_path, train_dir / class_name / file_path.name)
            
            for file_path in test_files:
                shutil.copy2(file_path, test_dir / class_name / file_path.name)
            
            print(f"Class '{class_name}': {len(train_files)} train, {len(test_files)} test images")
    
    def create_labels_file(self, data_dir, output_file='labels.json'):
        """
        Create a labels.json file for the dataset
        
        Args:
            data_dir: Directory containing organized data (train/test subdirectories)
            output_file: Output file name
        """
        data_path = Path(data_dir)
        labels = []
        
        # Process train and test directories
        for split in ['train', 'test']:
            split_dir = data_path / split
            if not split_dir.exists():
                continue
                
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                # Assign class index based on directory name
                if class_name == 'normal':
                    class_idx = 0
                elif class_name == 'mild':
                    class_idx = 1
                elif class_name == 'moderate':
                    class_idx = 2
                elif class_name == 'severe':
                    class_idx = 3
                elif class_name == 'paralysis':
                    class_idx = 1
                else:
                    class_idx = 0  # default
                
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # Relative path from data_dir
                        rel_path = img_file.relative_to(data_path)
                        labels.append({
                            'image': str(rel_path),
                            'label': class_idx,
                            'class_name': class_name,
                            'split': split
                        })
        
        # Save labels file
        output_path = data_path / output_file
        with open(output_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"Labels file created: {output_path}")
        print(f"Total samples: {len(labels)}")
        
        # Print statistics
        class_counts = {}
        for label in labels:
            class_name = label['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    
    def validate_dataset(self, data_dir):
        """
        Validate the organized dataset
        
        Args:
            data_dir: Directory containing organized data
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Error: Dataset directory {data_path} does not exist")
            return False
        
        # Check for train/test splits
        for split in ['train', 'test']:
            split_dir = data_path / split
            if not split_dir.exists():
                print(f"Warning: {split} directory not found")
                continue
            
            print(f"\n{split.upper()} SET:")
            total_images = 0
            
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                # Count images
                image_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_count += len(list(class_dir.glob(ext)))
                    image_count += len(list(class_dir.glob(ext.upper())))
                
                print(f"  {class_dir.name}: {image_count} images")
                total_images += image_count
            
            print(f"  Total: {total_images} images")
        
        return True


def main():
    """
    Example usage of the data preparator
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare facial paralysis dataset')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory for the dataset')
    parser.add_argument('--task_type', type=str, choices=['binary', 'multiclass'],
                       default='binary', help='Type of classification task')
    parser.add_argument('--normal_dir', type=str, help='Directory with normal images')
    parser.add_argument('--paralysis_dir', type=str, help='Directory with paralysis images')
    parser.add_argument('--mild_dir', type=str, help='Directory with mild paralysis images')
    parser.add_argument('--moderate_dir', type=str, help='Directory with moderate paralysis images')
    parser.add_argument('--severe_dir', type=str, help='Directory with severe paralysis images')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    preparator = FacialParalysisDataPreparator(args.root_dir)
    
    if args.task_type == 'binary':
        if not args.normal_dir or not args.paralysis_dir:
            print("For binary classification, --normal_dir and --paralysis_dir are required")
            return
        
        preparator.organize_binary_classification(
            normal_dir=args.normal_dir,
            paralysis_dir=args.paralysis_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
    elif args.task_type == 'multiclass':
        class_dirs = [args.normal_dir, args.mild_dir, args.moderate_dir, args.severe_dir]
        class_names = ['normal', 'mild', 'moderate', 'severe']
        
        # Filter out None values
        valid_pairs = [(dir_name, class_name) for dir_name, class_name in zip(class_dirs, class_names) if dir_name]
        
        if len(valid_pairs) < 2:
            print("For multiclass classification, at least 2 class directories are required")
            return
        
        class_dirs, class_names = zip(*valid_pairs)
        
        preparator.organize_multi_class_grading(
            class_dirs=class_dirs,
            class_names=class_names,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    
    # Create labels file
    preparator.create_labels_file(args.root_dir)
    
    # Validate dataset
    preparator.validate_dataset(args.root_dir)


if __name__ == '__main__':
    main()