# GPU Optimization Guide - RTX 3090 (24GB)

## 🚀 Optimizations Implemented

### 1. **Mixed Precision Training (AMP)**
- Uses FP16 for forward/backward passes
- Reduces memory usage by ~50%
- Increases throughput by ~2-3x on RTX 3090
- Automatic gradient scaling prevents underflow

### 2. **DataLoader Optimizations**
- **num_workers=8**: Parallel data loading (CPU cores)
- **persistent_workers=True**: Workers stay alive between epochs
- **prefetch_factor=4**: Loads 4 batches ahead
- **pin_memory=True**: Faster GPU transfers
- **non_blocking=True**: Async GPU transfers

### 3. **Memory Optimizations**
- `set_to_none=True` in optimizer (faster than zero_grad)
- Gradient accumulation ready (if needed for larger batches)
- Efficient tensor operations

### 4. **Data Augmentation**
- Random horizontal flips
- Color jittering (brightness, contrast, saturation, hue)
- Improves generalization without extra data

## 📊 Performance Expectations

### Before Optimization:
- GPU Usage: 50-60%
- Batch size: 32-64
- Throughput: ~100-150 samples/sec

### After Optimization:
- GPU Usage: **90-100%** ✅
- Batch size: **128-256** (with 24GB VRAM)
- Throughput: **300-500 samples/sec** (2-3x faster)
- Training time: **50% reduction**

## 🎯 Recommended Settings for RTX 3090

### For 300W-LP (Large Dataset - 104K samples)
```bash
# Maximum throughput
python train_cnn_model.py --dataset 300wlp --epochs 50 --batch-size 128

# Ultra-fast (if system allows)
python train_cnn_model.py --dataset 300wlp --epochs 50 --batch-size 256

# Memory safe (if system struggles)
python train_cnn_model.py --dataset 300wlp --epochs 50 --batch-size 96
```

### For BIWI (Smaller Dataset - 12K samples)
```bash
# Recommended
python train_cnn_model.py --dataset biwi --epochs 50 --batch-size 64

# Can go higher
python train_cnn_model.py --dataset biwi --epochs 50 --batch-size 128
```

## 🔍 Monitoring GPU Usage

### During Training:
```bash
# Terminal 1: Start training
python train_cnn_model.py --dataset 300wlp --epochs 50 --batch-size 128

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Look for:
# - GPU Utilization: Should be 90-100%
# - Memory Usage: ~18-22 GB (out of 24 GB)
# - Power Usage: ~300-350W (near TDP)
```

### Alternative Monitoring:
```bash
# Detailed stats
nvtop

# Or use Python
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
```

## ⚡ Troubleshooting

### GPU Still Not Fully Utilized?

#### Issue: CPU Bottleneck
**Symptoms**: GPU usage drops during data loading  
**Solution**: Increase num_workers
```python
# In trainer_cnn.py, increase to:
num_workers=12  # or even 16
```

#### Issue: Small Batch Size
**Symptoms**: GPU usage 50-70%  
**Solution**: Increase batch size
```bash
# Try larger batches
python train_cnn_model.py --dataset 300wlp --batch-size 192
```

#### Issue: Disk I/O Bottleneck
**Symptoms**: Workers waiting for disk reads  
**Solution**: 
- Use SSD instead of HDD
- Preload dataset to RAM disk
- Use image caching

### Out of Memory (OOM)?

If you get CUDA OOM errors:

```bash
# Reduce batch size
python train_cnn_model.py --dataset 300wlp --batch-size 64

# Or reduce workers
# Edit trainer_cnn.py: num_workers=4
```

### System Freezing?

If system becomes unresponsive:

```bash
# Reduce workers (too many CPU workers)
# Edit trainer_cnn.py: num_workers=4

# Limit CPU threads
export OMP_NUM_THREADS=4
python train_cnn_model.py --dataset 300wlp --batch-size 128
```

## 📈 Benchmark Results

### Expected Training Times (50 epochs)

| Dataset | Batch Size | Old (unoptimized) | New (optimized) | Speedup |
|---------|------------|-------------------|-----------------|---------|
| BIWI    | 64         | ~45 min          | ~20 min         | 2.25x   |
| 300W-LP | 128        | ~5 hours         | ~2.5 hours      | 2.0x    |
| 300W-LP | 256        | N/A (OOM)        | ~1.5 hours      | 3.3x    |

*Times are approximate and depend on CPU, SSD speed, etc.*

## 🎛️ Advanced Optimizations (Optional)

### 1. Gradient Accumulation (for very large effective batch sizes)
```python
# If you want effective batch_size=512 but GPU only fits 128
accumulation_steps = 4  # 128 * 4 = 512
```

### 2. Channels Last Memory Format
```python
# Faster on Ampere GPUs (RTX 30 series)
model = model.to(memory_format=torch.channels_last)
```

### 3. Compile Model (PyTorch 2.0+)
```python
model = torch.compile(model)  # ~10-20% speedup
```

### 4. TensorFloat-32 (TF32)
```python
# Already enabled by default on RTX 3090
# Provides speedup without code changes
torch.backends.cuda.matmul.allow_tf32 = True
```

## 🔧 System Optimization

### 1. CPU Governor
```bash
# Set CPU to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. GPU Persistence Mode
```bash
# Keep GPU initialized
sudo nvidia-smi -pm 1
```

### 3. GPU Clock Boost
```bash
# Lock GPU to max clocks (for consistent performance)
sudo nvidia-smi -lgc 1900  # Adjust based on your card
```

### 4. Disable Desktop Effects (if using GUI)
```bash
# Reduces background GPU usage
# Switch to TTY or use headless mode
```

## 📊 Validation

After implementing optimizations, you should see:

### Good Signs:
- ✅ GPU utilization: 90-100%
- ✅ GPU memory: 18-22 GB used
- ✅ Training speed: 300-500 samples/sec
- ✅ No CPU bottleneck warnings
- ✅ Consistent throughput across epochs

### Bad Signs:
- ❌ GPU utilization: <70%
- ❌ Frequent "CUDA out of memory" errors
- ❌ System freezing
- ❌ Slowing down after first epoch

## 🎯 Quick Test

Test the optimizations with a short run:

```bash
# Test run (5 epochs, large batch)
python train_cnn_model.py --dataset 300wlp --epochs 5 --batch-size 128

# Monitor in another terminal
nvidia-smi dmon -s u -d 1  # Update every second
```

Expected output:
```
# gpu   pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk
# Idx     W      C      C      %      %      %      %    MHz    MHz
    0   340     75      -     98     85      0      0   9501   1900
```

- `sm` (GPU utilization): Should be **>90%**
- `mem` (Memory utilization): Should be **>70%**
- `pwr` (Power): Should be near **350W** (TDP)

## 🚀 Ready to Train!

With all optimizations active, you're ready for high-speed training:

```bash
# Full 300W-LP training with maximum throughput
python train_cnn_model.py --dataset 300wlp --epochs 50 --batch-size 128
```

**Expected Results:**
- Training time: ~2-2.5 hours (down from ~5 hours)
- GPU utilization: 90-100%
- Model MAE: ~4-6° on 300W-LP

Happy training! 🎉
