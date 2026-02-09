"""
Training Script for Fusion Strategy Ablation Study

This script trains different fusion strategy variants:
1. Early Fusion: CNN → FC (no branch split)
2. Attention Fusion: CNN → Attention-weighted combination
3. Bilinear Pooling: CNN → Outer product fusion
4. Late Fusion: CNN → Separate encoding → Concatenation (baseline)

Features:
- Automatic data preprocessing (skips if already done)
- Fusion strategy ablation experiments
- Mixed precision training
- Early stopping
- Automatic result generation

Usage:
    python train_fusion_ablation.py --fusion_type early --epochs 150
    python train_fusion_ablation.py --fusion_type attention --epochs 150
    python train_fusion_ablation.py --fusion_type bilinear --epochs 150
    python train_fusion_ablation.py --fusion_type late --epochs 150
"""
import sys
sys.path.append('.')

import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.fusion_variants import create_fusion_variant, count_parameters
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


def train_fusion_variant(args):
    """Train a specific fusion strategy variant"""
    
    # Load config
    config = load_config()
    device = get_device(config)
    
    print("=" * 70)
    print(f"🔗 FUSION STRATEGY ABLATION: {args.fusion_type.upper()}")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Fusion Type: {args.fusion_type}")
    print(f"   Backbone:    {args.backbone}")
    print(f"   Face Dim:    {args.face_dim}")
    print(f"   Pose Dim:    {args.pose_dim}")
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
    model = create_fusion_variant(
        fusion_type=args.fusion_type,
        backbone=args.backbone,
        pretrained=True,
        output_dim=3,
        face_dim=args.face_dim,
        pose_dim=args.pose_dim
    ).to(device)
    
    params = count_parameters(model)
    print(f"   Fusion Type:  {args.fusion_type}")
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
    if args.backbone_ablation:
        save_path = output_dir / f"{args.fusion_type}_{args.backbone}_best.pth"
    elif args.dim_ablation:
        save_path = output_dir / f"{args.fusion_type}_{args.backbone}_best_{args.face_dim}:{args.pose_dim}.pth"
    else:
        save_path = output_dir / f"fusion_{args.fusion_type}_{args.backbone}_best.pth"
    
    # Training state
    best_val_mae = float('inf')
    best_val_metrics = None
    epochs_no_improve = 0
    history = []
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, angles in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
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
        
        # Log history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae_mean,
            'val_mae': val_mae_mean,
            'val_yaw': val_mae[0].item(),
            'val_pitch': val_mae[1].item(),
            'val_roll': val_mae[2].item()
        })
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train MAE: {train_mae_mean:.4f}° | "
                  f"Val MAE: {val_mae_mean:.4f}° "
                  f"(Yaw={val_mae[0]:.4f}°, Pitch={val_mae[1]:.4f}°, Roll={val_mae[2]:.4f}°)")
        
        # Save best model
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_val_metrics = {
                'yaw': val_mae[0].item(),
                'pitch': val_mae[1].item(),
                'roll': val_mae[2].item(),
                'mean': val_mae_mean
            }
            epochs_no_improve = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_mae': val_mae_mean,
                'val_metrics': best_val_metrics,
                'y_mean': y_mean_tensor.cpu(),
                'y_std': y_std_tensor.cpu(),
                'fusion_type': args.fusion_type,
                'backbone': args.backbone
            }, save_path)
            
            if epoch % 10 == 0:
                print(f"   ✓ New best model saved (Val MAE: {val_mae_mean:.4f}°)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break
    
    # Save training history
    history_df = pd.DataFrame(history)
    if args.backbone_ablation:
        history_path = output_dir / f"{args.fusion_type}_{args.backbone}_history.csv"
    elif args.dim_ablation:
        history_path = output_dir / f"{args.fusion_type}_{args.backbone}_history_{args.face_dim}:{args.pose_dim}.csv"
    else:
        history_path = output_dir / f"fusion_{args.fusion_type}_{args.backbone}_history.csv"
    history_df.to_csv(history_path, index=False)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Best Val MAE:     {best_val_mae:.4f}°")
    print(f"   Yaw MAE:          {best_val_metrics['yaw']:.4f}°")
    print(f"   Pitch MAE:        {best_val_metrics['pitch']:.4f}°")
    print(f"   Roll MAE:         {best_val_metrics['roll']:.4f}°")
    print(f"   Final Epoch:      {epoch}")
    print(f"   Model saved to:   {save_path}")
    print(f"   History saved to: {history_path}")
    print(f"   Parameters:       {params:,}")
    
    return best_val_metrics


def evaluate_on_test(args):
    """Evaluate trained model on test set"""
    
    config = load_config()
    device = get_device(config)
    
    print("\n" + "=" * 70)
    print(f"📊 EVALUATING {args.fusion_type.upper()} FUSION ON TEST SET")
    print("=" * 70)
    
    # Load model
    if args.backbone_ablation:
        model_path = Path(args.output_dir) / f"{args.fusion_type}_{args.backbone}_best.pth"
    elif args.dim_ablation:
        model_path = Path(args.output_dir) / f"{args.fusion_type}_{args.backbone}_best_{args.face_dim}:{args.pose_dim}.pth"
    else:
        model_path = Path(args.output_dir) / f"fusion_{args.fusion_type}_{args.backbone}_best.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = create_fusion_variant(
        fusion_type=args.fusion_type,
        backbone=args.backbone,
        pretrained=False,
        output_dim=3,
        face_dim=args.face_dim,
        pose_dim=args.pose_dim
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    y_mean = checkpoint['y_mean'].to(device)
    y_std = checkpoint['y_std'].to(device)
    
    # Load test data
    splits_path = config['data']['splits_path']
    test = np.load(os.path.join(splits_path, 'biwi_cnn_test.npz'), allow_pickle=True)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = BIWIImageDataset(
        test['image_paths'],
        test['labels'],
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Test samples: {len(test_dataset)}")
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, angles in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            angles = angles.to(device)
            
            with torch.amp.autocast('cuda'):
                predictions = model(images)
            
            # Denormalize
            predictions = predictions * y_std + y_mean
            angles = angles * y_std + y_mean
            
            all_preds.append(predictions.cpu())
            all_targets.append(angles.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mae = torch.abs(all_preds - all_targets).mean(dim=0)
    mae_mean = mae.mean().item()
    
    metrics = {
        'fusion_type': args.fusion_type,
        'yaw': mae[0].item(),
        'pitch': mae[1].item(),
        'roll': mae[2].item(),
        'mean': mae_mean,
        'n_samples': len(test_dataset)
    }
    
    print(f"\n📈 Test Results:")
    print(f"   Yaw MAE:   {metrics['yaw']:.4f}°")
    print(f"   Pitch MAE: {metrics['pitch']:.4f}°")
    print(f"   Roll MAE:  {metrics['roll']:.4f}°")
    print(f"   Mean MAE:  {metrics['mean']:.4f}°")
    
    # Save results
    results_dir = Path('results/csv_reports')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([metrics])
    if args.backbone_ablation:
        results_path = results_dir / f"backbone_{args.backbone}_test_results.csv"
    elif args.dim_ablation:
        results_path = results_dir / f"dim_attention_{args.face_dim}_{args.pose_dim}_test_results.csv"
    else:
        results_path = results_dir / f"fusion_{args.fusion_type}_test_results.csv"
    results_df.to_csv(results_path, index=False, float_format='%.4f')
    print(f"\n   Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train fusion strategy variants for ablation study'
    )
    
    parser.add_argument(
        '--fusion_type',
        type=str,
        required=True,
        choices=['early', 'attention', 'bilinear', 'late'],
        help='Fusion strategy to train'
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
        default='models/ablation/fusion',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate (skip training)'
    )
    
    parser.add_argument(
        '--face_dim',
        type=int,
        default=256,
        help='Face feature dimension (default: 256, use 1404 for geometry-guided)'
    )
    
    parser.add_argument(
        '--pose_dim',
        type=int,
        default=64,
        help='Pose feature dimension (default: 64, use 99 for geometry-guided)'
    )
    
    parser.add_argument(
        '--dim_ablation',
        action='store_true',
        help='Dimension ablation mode: include face_dim:pose_dim in model/result filenames'
    )
    
    parser.add_argument(
        '--backbone_ablation',
        action='store_true',
        help='Backbone ablation mode: save to backbone-specific folder with backbone in result filenames'
    )
    
    args = parser.parse_args()
    
    # If dim_ablation, default output_dir to dimension_attention folder
    if args.dim_ablation and args.output_dir == 'models/ablation/fusion':
        args.output_dir = 'models/ablation/dimension_attention'
    
    # If backbone_ablation, default output_dir to backbone folder
    if args.backbone_ablation and args.output_dir == 'models/ablation/fusion':
        args.output_dir = 'models/ablation/backbone'
    
    if not args.eval_only:
        # Train
        best_metrics = train_fusion_variant(args)
        print(f"\n✅ Fusion variant '{args.fusion_type}' training complete!")
        print(f"   Best validation MAE: {best_metrics['mean']:.4f}°")
    
    # Evaluate on test set
    test_metrics = evaluate_on_test(args)
    
    if test_metrics:
        print(f"\n✅ Test evaluation complete!")
        print(f"   Test MAE: {test_metrics['mean']:.4f}°")


if __name__ == '__main__':
    main()
