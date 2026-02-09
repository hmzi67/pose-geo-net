#!/bin/bash
# ======================================================================
# Dimension Ablation with Attention Fusion
# Runs all 5 dimension configs sequentially
# ======================================================================

set -e  # Exit on error

cd /home/genesys/hamza/cervical/Archive
source venv/bin/activate

echo "======================================================================"
echo "🚀 DIMENSION ABLATION (ATTENTION FUSION) - Starting all experiments"
echo "======================================================================"
echo ""

# 1) 512:128
echo "======================================================================"
echo "[1/5] Training attention fusion with dims 512:128"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 512 \
    --pose_dim 128 \
    --epochs 150 \
    --dim_ablation
echo ""

# 2) 1024:256
echo "======================================================================"
echo "[2/5] Training attention fusion with dims 1024:256"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 1024 \
    --pose_dim 256 \
    --epochs 150 \
    --dim_ablation
echo ""

# 3) 768:768
echo "======================================================================"
echo "[3/5] Training attention fusion with dims 768:768"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 768 \
    --pose_dim 768 \
    --epochs 150 \
    --dim_ablation
echo ""

# 4) 2048:64
echo "======================================================================"
echo "[4/5] Training attention fusion with dims 2048:64"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 2048 \
    --pose_dim 64 \
    --epochs 150 \
    --dim_ablation
echo ""

# 5) 1404:99
echo "======================================================================"
echo "[5/5] Training attention fusion with dims 1404:99"
echo "======================================================================"
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --dim_ablation
echo ""

echo "======================================================================"
echo "✅ ALL 5 DIMENSION ABLATION EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo "Results saved to: results/csv_reports/dim_attention_*_test_results.csv"
echo "Models saved to:  models/ablation/dimension_attention/"
