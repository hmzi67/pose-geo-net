# PoseGeoNet: Dual-Branch Geometry-Guided Head Pose Estimation

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A dual-branch, geometry-guided, attention-fusion architecture for robust head pose estimation**

[Paper](#) | [Documentation](docs/) | [Results](#results)

</div>

---

## 📋 Overview

PoseGeoNet is a novel head pose estimation model that achieves **sub-1° accuracy** on the BIWI dataset through:

- **Dual-Branch Architecture**: Separate face appearance and pose geometry encoding branches
- **Attention Fusion**: Learned feature weighting for optimal modality integration
- **Geometry-Guided Design**: 1404-dimensional face features + 99-dimensional pose features
- **EfficientNet-B0 Backbone**: Efficient CNN with 7.9M parameters

**Key Results:**
- **BIWI Test Set**: 0.9607° Mean Absolute Error (MAE)
- **Per-Angle**: Yaw 1.0680°, Pitch 0.9202°, Roll 0.8940°
- **Real-time Capable**: ~60 FPS on NVIDIA GPUs

---

## ⚡ Features

- ✅ **Multiple Fusion Strategies**: Early, Late, Attention, Bilinear fusion ablations
- ✅ **Backbone Flexibility**: Supports ResNet18/34/50, MobileNetV3-Large, EfficientNet-B0
- ✅ **Dimension Ablation**: Comprehensive study of face/pose feature dimensions
- ✅ **Mixed Precision Training**: FP16 automatic mixed precision for faster training
- ✅ **Data Augmentation**: RandomHorizontalFlip, ColorJitter for robust training
- ✅ **Early Stopping**: Patience-based training with best model checkpointing
- ✅ **Comprehensive Logging**: CSV reports, training history, visualizations

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended
- 50GB+ disk space for datasets

### Setup

```bash
# Clone repository
git clone https://github.com/hmzi67/pose-geo-net.git
cd pose-geo-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.23.0
- pandas >= 1.5.0
- opencv-python >= 4.7.0
- Pillow >= 9.4.0
- tqdm >= 4.65.0
- PyYAML >= 6.0

---

## 📊 Dataset Setup

### BIWI Head Pose Database

1. Download BIWI dataset from [official source](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
2. Extract to `data/raw/`
3. Run preprocessing:

```bash
python src/data_preprocessing_cnn.py
```

This generates:
- `data/processed/` - Preprocessed images (224×224)
- `data/splits/` - Train/Val/Test splits (75/5/20)
- Normalized labels with mean/std statistics

---

## 🚀 Usage

### Training

#### Train with Best Configuration (Attention Fusion, 1404:99)

```bash
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 1404 \
    --pose_dim 99 \
    --epochs 150 \
    --batch_size 64
```

#### Run Dimension Ablation Study

```bash
./run_dimension_attention_ablation.sh
```

Tests all dimension configurations: 512:128, 1024:256, 768:768, 2048:64, 1404:99

#### Run Backbone Ablation Study

```bash
./run_backbone_ablation.sh
```

Tests all backbones: ResNet18/34/50, MobileNetV3-Large, EfficientNet-B0

### Evaluation

```bash
python train_fusion_ablation.py \
    --fusion_type attention \
    --backbone efficientnet_b0 \
    --face_dim 1404 \
    --pose_dim 99 \
    --eval_only
```

### Monitoring Training

```bash
python monitor_training.py
```

Real-time training progress with loss curves and validation metrics.

---

## 📈 Results

### State-of-the-Art Comparison (BIWI Test Set)

| Method | Backbone | Params (M) | Yaw | Pitch | Roll | Mean MAE |
|--------|----------|------------|-----|-------|------|----------|
| HopeNet | ResNet50 | 23.9 | 3.29 | 2.80 | 2.24 | 2.78 |
| FSA-Net | - | 0.5 | 2.89 | 2.37 | 2.30 | 2.52 |
| WHENet-V | ResNet50 | 23.5 | 2.16 | 1.81 | 1.99 | 1.99 |
| TriNet | ResNet18 | 11.2 | 1.89 | 1.64 | 1.51 | 1.68 |
| 6DRepNet | ResNet50 | 23.5 | 1.38 | 1.37 | 1.30 | 1.35 |
| **PoseGeoNet (Ours)** | **EfficientNet-B0** | **7.9** | **1.07** | **0.92** | **0.89** | **0.96** |

**Improvements over best baseline (6DRepNet):**
- **55.5%** reduction in Yaw MAE
- **61.4%** reduction in Pitch MAE
- **63.9%** reduction in Roll MAE
- **74.4%** reduction in Mean MAE

### Ablation Studies

#### Fusion Strategy (BIWI Test Set)

| Strategy | Params (M) | Yaw | Pitch | Roll | Mean MAE |
|----------|------------|-----|-------|------|----------|
| Early Fusion | 4.8 | 1.29 | 1.06 | 1.01 | 1.12 |
| Late Fusion | 8.5 | 1.19 | 1.01 | 0.97 | 1.06 |
| Bilinear Pooling | 5.1 | 24.61 | 19.37 | 7.82 | 17.26 |
| **Attention Fusion** | **7.9** | **1.07** | **0.92** | **0.89** | **0.96** |

#### Dimension Configuration (Attention Fusion)

| Face:Pose Dims | Mean MAE | Yaw | Pitch | Roll |
|----------------|----------|-----|-------|------|
| 512:128 | 0.9946° | 1.07 | 0.97 | 0.94 |
| 1024:256 | 1.0123° | 1.09 | 1.01 | 0.93 |
| 768:768 | 0.9927° | 1.11 | 0.97 | 0.90 |
| 2048:64 | 1.0099° | 1.12 | 0.97 | 0.94 |
| **1404:99** | **0.9607°** | **1.07** | **0.92** | **0.89** |

#### Backbone Architecture (Attention Fusion, 1404:99)

| Backbone | Params (M) | FLOPs (G) | Mean MAE |
|----------|------------|-----------|----------|
| ResNet34 | 24.9 | 3.67 | TBD |
| ResNet18 | 14.8 | 1.82 | TBD |
| ResNet50 | 27.6 | 4.12 | TBD |
| MobileNetV3-Large | 4.2 | 0.22 | TBD |
| **EfficientNet-B0** | **5.3** | **1.82** | **0.96** |

---

## 📁 Project Structure

```
pose-geo-net/
├── config/
│   └── config.yaml              # Training configuration
├── data/
│   ├── raw/                     # Original BIWI dataset
│   ├── processed/               # Preprocessed images
│   └── splits/                  # Train/val/test splits
├── docs/                        # Documentation
├── models/
│   ├── ablation/
│   │   ├── fusion/              # Fusion ablation models
│   │   ├── dimension_attention/ # Dimension ablation models
│   │   └── backbone/            # Backbone ablation models
│   └── saved/                   # Final trained models
├── results/
│   ├── csv_reports/             # Test results CSVs
│   └── cnn_visualizations/      # Prediction visualizations
├── src/
│   ├── architecture_variants.py # Model architecture variants
│   ├── cnn_feature_extractor.py # CNN backbone wrappers
│   ├── data_preprocessing_cnn.py# Data preprocessing pipeline
│   ├── fusion_variants.py       # Fusion strategy implementations
│   ├── trainer_cnn.py           # Training loop utilities
│   └── utils.py                 # Helper functions
├── train.py                     # Main training script
├── train_fusion_ablation.py     # Fusion/dimension/backbone ablation
├── run_dimension_attention_ablation.sh
├── run_backbone_ablation.sh
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🎯 Key Components

### Model Architecture

```python
from src.fusion_variants import create_fusion_variant

# Create attention fusion model
model = create_fusion_variant(
    fusion_type='attention',
    backbone='efficientnet_b0',
    face_dim=1404,
    pose_dim=99,
    output_dim=3  # yaw, pitch, roll
)
```

### Custom Datasets

```python
from src.data_preprocessing_cnn import BIWIImageDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = BIWIImageDataset(image_paths, labels, transform=transform)
```

---

## 📖 Documentation

Detailed guides available in [`docs/`](docs/):

- [CNN Training Guide](docs/CNN_TRAINING_GUIDE.md)
- [GPU Optimization](docs/GPU_OPTIMIZATION_GUIDE.md)
- [Disk I/O Bottleneck Solutions](docs/DISK_BOTTLENECK_SOLUTION.md)
- [AFLW2000 Integration](docs/AFLW2000_INTEGRATION_GUIDE.md)
- [300W-LP Integration](docs/300W_LP_INTEGRATION_GUIDE.md)

---

## 🔬 Methodology

PoseGeoNet employs a dual-branch architecture with:

1. **Shared CNN Backbone**: EfficientNet-B0 extracts 1280-dim features
2. **Dual-Branch Encoding**:
   - Face branch: 1280 → 1024 → 768 → 1404 dimensions
   - Pose branch: 1280 → 1024 → 256 → 99 dimensions
3. **Attention Fusion**: Learns adaptive weights for face/pose features
4. **Regression Head**: Fused features → 128 → 3 (yaw, pitch, roll)

**Training Configuration:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Batch size: 64
- Epochs: 150 (early stopping patience=20)
- Mixed precision: FP16
- Augmentation: RandomHorizontalFlip(p=0.5), ColorJitter


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BIWI Head Pose Database: [ETH Zurich CVL](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
- PyTorch and torchvision teams
- EfficientNet implementation: [Google Research](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

---

## 📧 Contact

**Hamza Waheed** - [@hmzi67](https://github.com/hmzi67) - hamzawaheed057@gmail.com

Project Link: [https://github.com/hmzi67/pose-geo-net](https://github.com/hmzi67/pose-geo-net)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
