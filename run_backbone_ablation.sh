#!/bin/bash
# Backbone Architecture Ablation Study
# Trains attention fusion (1404:99) with 5 different backbones
# Results saved to: results/csv_reports/backbone_<name>_test_results.csv
# Models saved to: models/ablation/backbone/

echo "======================================================================"
echo "🚀 BACKBONE ABLATION (ATTENTION FUSION 1404:99) - Starting all experiments"
echo "======================================================================"

# 1. ResNet34
echo ""
echo "======================================================================"
echo "[1/5] Training with ResNet34"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone resnet34 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --backbone_ablation

# 2. ResNet18
echo ""
echo "======================================================================"
echo "[2/5] Training with ResNet18"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone resnet18 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --backbone_ablation

# 3. ResNet50
echo ""
echo "======================================================================"
echo "[3/5] Training with ResNet50"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone resnet50 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --backbone_ablation

# 4. MobileNetV3-Large
echo ""
echo "======================================================================"
echo "[4/5] Training with MobileNetV3-Large"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone mobilenet_v3_large \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --backbone_ablation

# 5. EfficientNet-B0
echo ""
echo "======================================================================"
echo "[5/5] Training with EfficientNet-B0"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --backbone_ablation

echo ""
echo "======================================================================"
echo "✅ ALL BACKBONE ABLATION EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  results/csv_reports/backbone_resnet34_test_results.csv"
echo "  results/csv_reports/backbone_resnet18_test_results.csv"
echo "  results/csv_reports/backbone_resnet50_test_results.csv"
echo "  results/csv_reports/backbone_mobilenet_v3_large_test_results.csv"
echo "  results/csv_reports/backbone_efficientnet_b0_test_results.csv"
echo ""
echo "Models saved to: models/ablation/backbone/"
