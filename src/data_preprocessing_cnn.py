"""
Data preprocessing for CNN-based approach
Loads images directly instead of extracting landmarks
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.utils import read_pose_txt, rotation_matrix_to_euler, ensure_dir


class BIWIImageDataset(Dataset):
    """
    PyTorch Dataset for BIWI with raw images (CNN-based approach)
    Optimized for fast loading on RTX 3090
    """
    
    def __init__(self, image_paths, labels, transform=None, image_size=224):
        """
        Args:
            image_paths: List of image file paths
            labels: Numpy array of labels [N, 3] (yaw, pitch, roll)
            transform: Optional transform to apply
            image_size: Target image size (default 224 for ResNet)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        
        if transform is None:
            # Optimized transforms with aggressive augmentation
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image - using cv2 with IMREAD_COLOR flag for speed
        img_path = self.image_paths[idx]
        
        # Faster image loading with cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            # Fallback for corrupted images
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB inline
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transform - keep as numpy array if using transforms that expect it
        if self.transform:
            # Check if transform expects PIL Image or numpy array
            # Most torchvision transforms accept both, but we pass PIL to be safe
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            # No transform - just convert to PIL and then tensor
            img = Image.fromarray(img)
            img = transforms.ToTensor()(img)
        
        # Get label - avoid creating new tensor each time
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, label


class BIWIDatasetProcessorCNN:
    """Process BIWI Head Pose Database for CNN-based approach"""
    
    def __init__(self, config):
        self.config = config
        self.raw_data_path = config['data']['raw_data_path']
        self.processed_path = config['data']['processed_path']
        self.max_frames = config['data'].get('max_frames_per_subject')
        
        ensure_dir(self.processed_path)
    
    def collect_image_label_pairs(self):
        """
        Collect all image paths and labels (no feature extraction)
        """
        subjects = sorted(os.listdir(self.raw_data_path))
        
        print(f"Found {len(subjects)} subjects")
        
        all_image_paths = []
        all_labels = []
        
        for subj in subjects:
            subj_path = os.path.join(self.raw_data_path, subj)
            
            if not os.path.isdir(subj_path):
                continue
            
            frames = sorted([f for f in os.listdir(subj_path) if f.endswith("_rgb.png")])
            
            count = 0
            for fname in tqdm(frames, desc=f"Subject {subj}"):
                if self.max_frames and count >= self.max_frames:
                    break
                
                rgb_path = os.path.join(subj_path, fname)
                pose_path = rgb_path.replace("_rgb.png", "_pose.txt")
                
                if not os.path.exists(pose_path):
                    continue
                
                # Check if image can be loaded
                img = cv2.imread(rgb_path)
                if img is None:
                    continue
                
                # Read label
                try:
                    R, t = read_pose_txt(pose_path)
                    ypr = rotation_matrix_to_euler(R)
                except Exception:
                    continue
                
                all_image_paths.append(rgb_path)
                all_labels.append(ypr)
                count += 1
        
        all_labels = np.array(all_labels, dtype=np.float32)
        
        print(f"\nTotal samples collected: {len(all_image_paths)}")
        print(f"Labels shape: {all_labels.shape}")
        
        # Save paths and labels
        combined_path = os.path.join(self.processed_path, "biwi_cnn_data.npz")
        np.savez_compressed(
            combined_path,
            image_paths=np.array(all_image_paths),
            labels=all_labels
        )
        
        print(f"Saved to: {combined_path}")
        return combined_path
    
    def create_train_val_test_splits(self, combined_data_path):
        """Create train/val/test splits for CNN approach"""
        splits_path = self.config['data']['splits_path']
        ensure_dir(splits_path)
        
        # Load combined data
        data = np.load(combined_data_path, allow_pickle=True)
        image_paths = data["image_paths"]
        y = data["labels"]
        
        print(f"Loaded dataset: {len(image_paths)} images, {y.shape} labels")
        
        # Label normalization
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0) + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Split configuration
        split_cfg = self.config['split']
        test_ratio = split_cfg['test_ratio']
        val_ratio = split_cfg['val_ratio']
        random_state = split_cfg['random_state']
        
        # First split: separate test set
        paths_temp, paths_test, y_temp, y_test = train_test_split(
            image_paths, y_norm, 
            test_size=test_ratio, 
            random_state=random_state, 
            shuffle=True
        )
        
        # Second split: train and validation
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        paths_train, paths_val, y_train, y_val = train_test_split(
            paths_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            shuffle=True
        )
        
        # Save splits
        np.savez_compressed(
            os.path.join(splits_path, "biwi_cnn_train.npz"),
            image_paths=paths_train, labels=y_train,
            y_mean=y_mean, y_std=y_std
        )
        
        np.savez_compressed(
            os.path.join(splits_path, "biwi_cnn_val.npz"),
            image_paths=paths_val, labels=y_val,
            y_mean=y_mean, y_std=y_std
        )
        
        np.savez_compressed(
            os.path.join(splits_path, "biwi_cnn_test.npz"),
            image_paths=paths_test, labels=y_test,
            y_mean=y_mean, y_std=y_std
        )
        
        print(f"Splits saved to: {splits_path}")
        print(f"Train: {len(paths_train)} | Val: {len(paths_val)} | Test: {len(paths_test)}")
        
        return splits_path


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Preprocess BIWI dataset for CNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = BIWIDatasetProcessorCNN(config)
    
    print("="*60)
    print("CNN-BASED DATA PREPROCESSING")
    print("="*60)
    print("STEP 1: Collecting image paths and labels")
    print("="*60)
    combined_path = processor.collect_image_label_pairs()
    
    print("\n" + "="*60)
    print("STEP 2: Creating train/val/test splits")
    print("="*60)
    splits_path = processor.create_train_val_test_splits(combined_path)
    
    print("\n" + "="*60)
    print("✅ CNN DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Splits are ready in: {splits_path}")
    print("You can now proceed to training with CNN-based model.")
    print("="*60)
