#!/usr/bin/env python3
"""
Generate comprehensive accuracy threshold matrix for fusion strategy ablation.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os

import sys
sys.path.insert(0, '/home/genesys/hamza/cervical/Archive')

from src.utils import load_config, get_device
from src.data_preprocessing_cnn import BIWIImageDataset
from src.fusion_variants import create_fusion_variant

config = load_config()
device = get_device(config)

# Load test data
test_data = np.load('data/splits/biwi_cnn_test.npz', allow_pickle=True)

# Image preprocessing transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = BIWIImageDataset(
    test_data['image_paths'],
    test_data['labels'],
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def compute_accuracies(model, loader, y_mean, y_std):
    """Compute accuracy metrics at different thresholds."""
    model.eval()
    all_max_err = []
    
    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)
            
            preds = model(images)
            preds = preds * y_std + y_mean
            angles = angles * y_std + y_mean
            
            err = (preds - angles).abs()
            max_err = err.max(dim=1).values
            all_max_err.append(max_err.cpu())
    
    all_max_err = torch.cat(all_max_err)
    
    # Collect all errors for per-angle computation
    all_err = []
    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)
            
            preds = model(images)
            preds = preds * y_std + y_mean
            angles = angles * y_std + y_mean
            
            err = (preds - angles).abs()
            all_err.append(err.cpu())
    
    all_errors = torch.cat(all_err, dim=0)
    
    # Compute per-angle accuracies
    results = {}
    for angle_idx, angle_name in enumerate(['Yaw', 'Pitch', 'Roll']):
        angle_errors = all_errors[:, angle_idx]
        results[angle_name] = {}
        for threshold in [1, 2, 5]:
            acc = (angle_errors <= threshold).float().mean() * 100
            results[angle_name][f'Acc@{threshold}°'] = acc.item()
    
    # Compute overall accuracy
    results['Overall'] = {}
    for threshold in [1, 2, 5]:
        acc = (all_max_err <= threshold).float().mean().item() * 100
        results['Overall'][f'Acc@{threshold}°'] = acc
    
    return results

# Configuration for fusion strategy ablation
fusion_configs = {
    'models/ablation/fusion/fusion_early_efficientnet_b0_best.pth': ('early', 'Early Fusion'),
    'models/ablation/fusion/fusion_late_efficientnet_b0_best.pth': ('late', 'Late Fusion'),
    'models/ablation/fusion/fusion_attention_efficientnet_b0_best.pth': ('attention', 'Attention Fusion'),
    'models/ablation/fusion/fusion_bilinear_efficientnet_b0_best.pth': ('bilinear', 'Bilinear Fusion'),
}

results_dict = {}

print("\n" + "=" * 80)
print("FUSION STRATEGY ABLATION - Accuracy at Different Error Thresholds")
print("(All with EfficientNet-B0 backbone)")
print("=" * 80)

for checkpoint_path, (fusion_type, label) in fusion_configs.items():
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠ Checkpoint not found: {checkpoint_path}")
        continue
    
    print(f"\nProcessing: {label}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = create_fusion_variant(
            fusion_type=fusion_type,
            backbone='efficientnet_b0',
            pretrained=False,
            output_dim=3,
            face_dim=1404,
            pose_dim=99
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state'])
        
        y_mean = checkpoint['y_mean'].to(device)
        y_std = checkpoint['y_std'].to(device)
        
        # Compute accuracies
        print(f"  Computing accuracies...")
        results = compute_accuracies(model, test_loader, y_mean, y_std)
        results_dict[label] = results
        
        # Print results
        for threshold in [1, 2, 5]:
            y = results['Yaw'][f'Acc@{threshold}°']
            p = results['Pitch'][f'Acc@{threshold}°']
            r = results['Roll'][f'Acc@{threshold}°']
            o = results['Overall'][f'Acc@{threshold}°']
            print(f"  Acc@{threshold}°: Yaw={y:.2f}% | Pitch={p:.2f}% | Roll={r:.2f}% | Overall={o:.2f}%")
        
    except Exception as e:
        print(f"  ✗ Error processing checkpoint: {str(e)[:100]}")
        continue

# Generate formatted table output
print("\n" + "=" * 80)
print("FORMATTED TABLE OUTPUT")
print("=" * 80)

print("\nTABLE: Accuracy at different error thresholds for fusion strategy ablation")
print("(EfficientNet-B0 backbone)\n")

fusion_order = ['Early Fusion', 'Late Fusion', 'Attention Fusion', 'Bilinear Fusion']

for threshold in [1, 2, 5]:
    print(f"\nAcc@{threshold}°:")
    print("─" * 80)
    print(f"{'Fusion Strategy':<25} {'Yaw (%)':<15} {'Pitch (%)':<15} {'Roll (%)':<15} {'Overall (%)':<15}")
    print("─" * 80)
    
    for label in fusion_order:
        if label in results_dict:
            y = results_dict[label]['Yaw'][f'Acc@{threshold}°']
            p = results_dict[label]['Pitch'][f'Acc@{threshold}°']
            r = results_dict[label]['Roll'][f'Acc@{threshold}°']
            o = results_dict[label]['Overall'][f'Acc@{threshold}°']
            print(f"{label:<25} {y:>13.2f}  {p:>13.2f}  {r:>13.2f}  {o:>13.2f}")
    print()

# Generate LaTeX table
print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80)

print("\n\\begin{table}[htp]")
print("\\centering")
print("\\caption{Accuracy of PoseGeoNet with different fusion strategies at")
print("various error thresholds. All models use EfficientNet-B0 backbone with")
print("(face\_dim=1404, pose\_dim=99) on BIWI test set.}")
print("\\label{table:fusion-accuracy}")
print("\\begin{tabular}{ccccc}")
print("\\hline")
print("\\textbf{Fusion Strategy} & \\textbf{Yaw} & \\textbf{Pitch} & \\textbf{Roll} & \\textbf{Overall} \\\\")
print("\\hline")

for threshold in [1, 2, 5]:
    print(f"\\multicolumn{{5}}{{c}}{{\\textbf{{Acc@{threshold}°}}}} \\\\")
    print("\\hline")
    for label in fusion_order:
        if label in results_dict:
            y = results_dict[label]['Yaw'][f'Acc@{threshold}°']
            p = results_dict[label]['Pitch'][f'Acc@{threshold}°']
            r = results_dict[label]['Roll'][f'Acc@{threshold}°']
            o = results_dict[label]['Overall'][f'Acc@{threshold}°']
            print(f"{label} & {y:.2f}\\% & {p:.2f}\\% & {r:.2f}\\% & {o:.2f}\\% \\\\")
    if threshold < 5:
        print("\\hline")

print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

# Save to JSON
output_file = 'accuracy_matrix_fusion_results.json'
with open(output_file, 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n" + "=" * 80)
print(f"✓ Results saved to: {output_file}")
print("=" * 80 + "\n")
