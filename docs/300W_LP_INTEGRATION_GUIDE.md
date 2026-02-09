# 300W-LP Dataset Integration Guide

## Overview

The **300W-LP** (300 Faces in-the-wild Large Pose) dataset is a large-scale synthetic face dataset with precise head pose annotations. It contains **122,414 samples** with diverse head poses, making it excellent for training robust head pose estimation models.

## Dataset Structure

### What You Have:
```
data/300W_LP/
├── 300wlp_list.txt          # Full dataset list (122,414 samples)
├── val_small_list.txt       # Small validation set (63 samples)
├── AFW/                     # Subset from AFW dataset
├── AFW_Flip/                # Horizontally flipped AFW images
├── HELEN/                   # Subset from HELEN dataset
├── HELEN_Flip/              # Flipped HELEN images
├── IBUG/                    # Subset from IBUG dataset
├── IBUG_Flip/               # Flipped IBUG images
├── LFPW/                    # Subset from LFPW dataset
└── LFPW_Flip/               # Flipped LFPW images
```

### Data Format:
Each sample consists of:
- **Image**: `*.jpg` files (face crops)
- **Annotations**: `*.mat` MATLAB files with:
  - `Pose_Para` (1, 7): **[pitch, yaw, roll, tx, ty, scale, tz]**
    - First 3 values are **rotation in radians**
    - Remaining values are for 3DMM fitting parameters
  - `pt2d` (2, 68): 2D facial landmarks (68 points)
  - `roi` (1, 4): Bounding box
  - Additional 3DMM parameters (Illum_Para, Color_Para, etc.)

## Key Differences from BIWI Dataset

| Aspect | BIWI | 300W-LP |
|--------|------|---------|
| Size | ~15,000 frames | 122,414 samples |
| Source | Real depth sensor | Synthetic (3DMM) |
| Pose Range | Natural poses | Extreme poses (-90° to +90°) |
| Image Type | RGB + Depth | RGB only |
| Annotation | _pose.txt files | .mat files |
| Angles Format | Degrees | Radians |
| File Structure | frame_XX_rgb.png | dataset_name_id.jpg |

## Integration Steps

### Step 1: Create 300W-LP Data Preprocessor

I've created `src/data_preprocessing_300wlp.py` that:
- Reads .mat files and extracts pose annotations
- Converts radians to degrees
- Loads images and applies same preprocessing as BIWI
- Handles the train/val split using provided lists
- Maintains compatibility with existing training pipeline

### Step 2: Update Configuration

Add to `config/config.yaml`:
```yaml
data:
  # Existing BIWI paths
  raw_data_path: "data/raw/faces_0"
  
  # New 300W-LP paths
  dataset_300wlp_path: "data/300W_LP"
  dataset_300wlp_train_list: "data/300W_LP/300wlp_list.txt"
  dataset_300wlp_val_list: "data/300W_LP/val_small_list.txt"
```

### Step 3: Preprocess 300W-LP Data

Run the preprocessing script:
```bash
python preprocess_300wlp_data.py
```

This will:
- Create `data/splits/300wlp_cnn_train.npz`
- Create `data/splits/300wlp_cnn_val.npz`
- Create `data/splits/300wlp_cnn_test.npz` (optional split from train)

### Step 4: Train on 300W-LP

Use the training script with 300W-LP flag:
```bash
python train_cnn_model.py --dataset 300wlp
```

Or train on both datasets:
```bash
python train_cnn_model.py --dataset combined
```

### Step 5: Compare Results

After training on both datasets, compare:
```bash
python compare_datasets.py
```

## Expected Performance Improvements

### Advantages of 300W-LP:
1. **8x more data** → Better generalization
2. **Extreme poses** → Better handling of large angles
3. **Diverse faces** → Better facial appearance variations
4. **Data augmentation** → Built-in flipped versions

### Potential Challenges:
1. **Synthetic data** → May not capture real-world variations
2. **Domain gap** → Rendered faces vs. real images
3. **Annotation noise** → 3DMM fitting errors

## Training Strategies

### Option 1: Train Separately (Recommended for Comparison)
```bash
# Train on BIWI only
python train_cnn_model.py --dataset biwi

# Train on 300W-LP only
python train_cnn_model.py --dataset 300wlp

# Compare results
python compare_datasets.py
```

### Option 2: Combined Training
```bash
# Train on both datasets
python train_cnn_model.py --dataset combined
```

### Option 3: Transfer Learning
```bash
# Pre-train on 300W-LP (larger dataset)
python train_cnn_model.py --dataset 300wlp --save-path models/saved/pretrained_300wlp.pth

# Fine-tune on BIWI (target domain)
python train_cnn_model.py --dataset biwi --load-path models/saved/pretrained_300wlp.pth
```

## Implementation Files

I'm creating the following files for you:

1. **`src/data_preprocessing_300wlp.py`**
   - Data loader for 300W-LP dataset
   - Handles .mat file parsing
   - Creates train/val/test splits

2. **`preprocess_300wlp_data.py`**
   - Standalone script to preprocess 300W-LP
   - Creates .npz files compatible with training pipeline

3. **`train_cnn_model_updated.py`** (updated version)
   - Supports `--dataset` flag: `biwi`, `300wlp`, or `combined`
   - Automatic dataset loading based on flag

4. **`compare_datasets.py`**
   - Compares training results on both datasets
   - Generates comparison plots and metrics

## Quick Start

```bash
# 1. Preprocess 300W-LP data
python preprocess_300wlp_data.py

# 2. Train on 300W-LP
python train_cnn_model.py --dataset 300wlp --epochs 100

# 3. Compare with BIWI results
python compare_datasets.py --biwi-model models/saved/cervical_headpose_cnn_best.pth \
                           --300wlp-model models/saved/300wlp_headpose_cnn_best.pth
```

## Expected Results

Based on literature and the dataset characteristics:

| Metric | BIWI (Current) | 300W-LP (Expected) | Combined (Expected) |
|--------|----------------|-------------------|---------------------|
| MAE (degrees) | ~5-7° | ~4-6° | ~4-5° |
| Training Time | Fast (~1-2h) | Slower (~8-10h) | Longest (~10-12h) |
| Generalization | Good on frontal | Good on extreme | Best overall |
| Real-world | Excellent | Good | Excellent |

## Next Steps

1. Run preprocessing to create splits
2. Train on 300W-LP and compare with BIWI
3. Experiment with combined training
4. Try transfer learning (pre-train on 300W-LP, fine-tune on BIWI)
5. Evaluate on both test sets to assess cross-dataset performance
