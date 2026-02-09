"""
Script: Train models with different backbones for ablation study
Usage: python train_backbone_ablation.py --backbone resnet34

Supported backbones:
- resnet18 (baseline)
- resnet34
- resnet50
- mobilenet_v3_large
- mobilenet_v3_small
- efficientnet_b0
"""
import sys
sys.path.append('.')

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils import load_config, get_device, ensure_dir
from src.cnn_feature_extractor import create_cnn_based_model
from src.trainer_cnn import CNNHeadPoseTrainer
from torch.utils.data import DataLoader
from torchvision import transforms


def get_backbone_config(backbone_name, base_config):
    """Create config with specified backbone"""
    config = base_config.copy()
    config['model']['backbone'] = backbone_name
    return config


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_backbone_variant(backbone, base_config, output_dir):
    """Train model with specified backbone"""
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING: {backbone.upper()}")
    print("="*70)
    
    # Update config with backbone
    config = get_backbone_config(backbone, base_config)
    device = get_device(config)
    
    # Create model to count parameters
    print(f"\n1. Creating model with {backbone} backbone...")
    temp_model = create_cnn_based_model(config).to(device)
    n_params = count_parameters(temp_model)
    print(f"   ✅ Model created: {n_params:,} trainable parameters")
    del temp_model  # Free memory
    torch.cuda.empty_cache()
    
    # Create trainer (it will create the model internally)
    print("\n2. Setting up trainer...")
    trainer = CNNHeadPoseTrainer(config, device, dataset='biwi')
    
    # Load data
    print("\n3. Loading data...")
    splits_path = config['data']['splits_path']
    trainer.load_data(splits_path)
    
    # Training
    print(f"\n4. Training for {config['training']['epochs']} epochs...")
    print(f"   💾 Checkpoints will be saved to: {output_dir}")
    
    # Modify save directory in trainer
    original_save_dir = config['model'].get('save_dir', 'models/saved')
    
    # Train
    trainer.train()
    
    # Get final results
    history = trainer.history
    best_val_mae = trainer.best_val_mae
    
    # Save best model to custom directory with backbone name
    best_model_path = output_dir / f"{backbone}_best.pth"
    torch.save({
        'model_state': trainer.model.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict(),
        'epoch': len(history['epoch']),
        'best_val_mae': best_val_mae,
        'y_mean': trainer.y_mean,
        'y_std': trainer.y_std,
        'config': config,
        'backbone': backbone
    }, best_model_path)
    
    print(f"\n✅ Training complete for {backbone}!")
    print(f"   📊 Best validation MAE: {best_val_mae:.4f}°")
    print(f"   💾 Model saved to: {best_model_path}")
    
    # Save metadata
    metadata = {
        'backbone': backbone,
        'n_parameters': n_params,
        'best_val_mae': float(best_val_mae),
        'epochs_trained': len(history['epoch']),
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open(output_dir / f"{backbone}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return history, metadata


def main():
    parser = argparse.ArgumentParser(description='Train backbone ablation experiments')
    parser.add_argument('--backbone', type=str, default='resnet34',
                       choices=['resnet18', 'resnet34', 'resnet50', 
                               'mobilenet_v3_large', 'mobilenet_v3_small', 
                               'efficientnet_b0'],
                       help='Backbone architecture to train')
    parser.add_argument('--output_dir', type=str, default='models/ablation/backbone',
                       help='Directory to save trained models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (optional)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size (optional)')
    
    args = parser.parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Override if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    print("="*70)
    print("🧪 BACKBONE ABLATION STUDY")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Backbone: {args.backbone}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Output dir: {output_dir}")
    
    # Train
    history, metadata = train_backbone_variant(args.backbone, config, output_dir)
    
    print("\n" + "="*70)
    print("✅ ABLATION EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\n📊 Results for {args.backbone}:")
    print(f"   Parameters: {metadata['n_parameters']:,}")
    print(f"   Best Val MAE: {metadata['best_val_mae']:.4f}°")
    print(f"\n💡 Next steps:")
    print(f"   1. Generate CSV results:")
    print(f"      python generate_results_csv.py \\")
    print(f"        --model_path {output_dir}/{args.backbone}_best.pth \\")
    print(f"        --experiment_name {args.backbone}")
    print(f"\n   2. Train other backbones:")
    print(f"      python train_backbone_ablation.py --backbone resnet50")
    print(f"      python train_backbone_ablation.py --backbone mobilenet_v3_large")


if __name__ == "__main__":
    main()
