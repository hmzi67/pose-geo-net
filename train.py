"""
Training Script for Architecture Ablation Study

This script trains different architectural variants:
1. Single-branch Direct: CNN → FC (minimal)
2. Single-branch MLP: CNN → MLP (more layers, no split)
3. Dual-branch: CNN → Face/Pose branches (current approach)

Features:
- Automatic data preprocessing (skips if already done)
- Architecture ablation experiments
- Mixed precision training
- Early stopping

Usage:
    python train.py --variant single_direct --epochs 150
    python train.py --variant single_mlp --epochs 150
    python train.py --variant dual_branch --epochs 150
"""
import sys
sys.path.append('.')

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from src.architecture_variants import create_architecture_variant, count_parameters
from src.data_preprocessing_cnn import BIWIImageDataset, BIWIDatasetProcessorCNN
from src.utils import load_config, get_device, ensure_dir


def check_and_preprocess_data(config):
    """
    Check if data preprocessing is complete, run if needed
    
    Returns:
        str: Path to splits directory
    """
    splits_path = config['data']['splits_path']
    
    # Check if all required split files exist
    required_files = [
        'biwi_cnn_train.npz',
        'biwi_cnn_val.npz', 
        'biwi_cnn_test.npz'
    ]
    
    all_exist = all(
        os.path.exists(os.path.join(splits_path, f)) 
        for f in required_files
    )
    
    if all_exist:
        print("✅ Data preprocessing already complete - skipping")
        print(f"   Using splits from: {splits_path}")
        return splits_path
    
    print("🔄 Data preprocessing required - starting...")
    print("=" * 70)
    
    # Ensure directories exist
    ensure_dir(config['data']['processed_path'])
    ensure_dir(splits_path)
    
    # Initialize processor
    processor = BIWIDatasetProcessorCNN(config)
    
    # Step 1: Check if combined data exists
    combined_path = os.path.join(config['data']['processed_path'], "biwi_cnn_data.npz")
    
    if not os.path.exists(combined_path):
        print("📸 Step 1: Collecting image paths and labels...")
        combined_path = processor.collect_image_label_pairs()
    else:
        print("✅ Step 1: Combined data already exists - skipping collection")
    
    # Step 2: Create splits
    print("📊 Step 2: Creating train/val/test splits...")
    splits_path = processor.create_train_val_test_splits(combined_path)
    
    print("✅ Data preprocessing complete!")
    print("=" * 70)
    
    return splits_path


def train_architecture_variant(args):
    """Train a specific architecture variant"""
    
    # Load config
    config = load_config()
    device = get_device(config)
    
    print("=" * 70)
    print(f"🏗️  ARCHITECTURE ABLATION: {args.variant.upper()}")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Variant:     {args.variant}")
    print(f"   Backbone:    {args.backbone}")
    print(f"   Epochs:      {args.epochs}")
    print(f"   Batch size:  {args.batch_size}")
    print(f"   Output dir:  {args.output_dir}")
    print(f"   Device:      {device}")
    
    # Check and preprocess data if needed
    print(f"\n🔍 Checking data preprocessing...")
    splits_path = check_and_preprocess_data(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data splits
    print("\n📂 Loading data splits...")
    train = np.load(os.path.join(splits_path, 'biwi_cnn_train.npz'), allow_pickle=True)
    val = np.load(os.path.join(splits_path, 'biwi_cnn_val.npz'), allow_pickle=True)
    
    print(f"   Train samples: {len(train['image_paths'])}")
    print(f"   Val samples:   {len(val['image_paths'])}")
    
    # Data normalization parameters
    y_mean = train['y_mean']
    y_std = train['y_std']
    y_mean_tensor = torch.tensor(y_mean, dtype=torch.float32).to(device)
    y_std_tensor = torch.tensor(y_std, dtype=torch.float32).to(device)
    
    print(f"   Y mean: {y_mean}")
    print(f"   Y std:  {y_std}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test should NOT have augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BIWIImageDataset(
        train['image_paths'],
        train['labels'],
        transform=train_transform
    )
    
    val_dataset = BIWIImageDataset(
        val['image_paths'],
        val['labels'],
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_architecture_variant(
        variant_type=args.variant,
        backbone=args.backbone,
        pretrained=True,
        output_dim=3
    ).to(device)
    
    params = count_parameters(model)
    print(f"   Architecture: {args.variant}")
    print(f"   Backbone:     {args.backbone}")
    print(f"   Parameters:   {params:,}")
    print(f"   Model size:   {params * 4 / (1024**2):.2f} MB")
    
    # Setup optimizer and loss
    print("\n⚙️  Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Model save path
    save_path = output_dir / f"{args.variant}_{args.backbone}_best.pth"
    
    # Training state
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, angles in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, angles)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            train_preds.append(predictions.detach().cpu())
            train_targets.append(angles.detach().cpu())
        
        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_preds = torch.cat(train_preds, dim=0) * y_std_tensor.cpu() + y_mean_tensor.cpu()
        train_targets = torch.cat(train_targets, dim=0) * y_std_tensor.cpu() + y_mean_tensor.cpu()
        train_mae = torch.abs(train_preds - train_targets).mean(dim=0)
        train_mae_mean = train_mae.mean().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, angles in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    loss = criterion(predictions, angles)
                
                val_loss += loss.item() * images.size(0)
                val_preds.append(predictions.cpu())
                val_targets.append(angles.cpu())
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_preds = torch.cat(val_preds, dim=0) * y_std_tensor.cpu() + y_mean_tensor.cpu()
        val_targets = torch.cat(val_targets, dim=0) * y_std_tensor.cpu() + y_mean_tensor.cpu()
        val_mae = torch.abs(val_preds - val_targets).mean(dim=0)
        val_mae_mean = val_mae.mean().item()
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train MAE: {train_mae_mean:.4f}° | "
                  f"Val MAE: {val_mae_mean:.4f}° "
                  f"(Yaw={val_mae[0]:.4f}°, Pitch={val_mae[1]:.4f}°, Roll={val_mae[2]:.4f}°)")
        
        # Save best model
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            epochs_no_improve = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_mae': val_mae_mean,
                'y_mean': y_mean_tensor.cpu(),
                'y_std': y_std_tensor.cpu()
            }, save_path)
            
            if epoch % 10 == 0:
                print(f"   ✓ New best model saved (Val MAE: {val_mae_mean:.4f}°)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Best Val MAE:     {best_val_mae:.4f}°")
    print(f"   Final Epoch:      {epoch}")
    print(f"   Model saved to:   {save_path}")
    print(f"   Parameters:       {params:,}")
    
    return best_val_mae


def main():
    parser = argparse.ArgumentParser(
        description='Train architecture variants for ablation study'
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        required=True,
        choices=['single_direct', 'single_mlp', 'dual_branch'],
        help='Architecture variant to train'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v3_large', 'efficientnet_b0'],
        help='CNN backbone architecture'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/ablation/architecture',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Train
    best_mae = train_architecture_variant(args)
    
    print(f"\n✅ Architecture variant '{args.variant}' training complete!")
    print(f"   Best validation MAE: {best_mae:.4f}°")


if __name__ == '__main__':
    main()
