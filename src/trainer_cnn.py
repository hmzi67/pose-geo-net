"""
Trainer for CNN-based head pose estimation
"""
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cnn_feature_extractor import create_cnn_based_model
from src.data_preprocessing_cnn import BIWIImageDataset
from src.utils import ensure_dir


class CNNHeadPoseTrainer:
    """Trainer for CNN-based head pose estimation model"""
    
    def __init__(self, config, device, dataset='biwi'):
        self.config = config
        self.device = device
        self.dataset = dataset  # 'biwi' or '300wlp'
        
        # Training params
        train_cfg = config['training']
        self.batch_size = train_cfg['batch_size']
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.patience = train_cfg['early_stopping_patience']
        self.log_interval = train_cfg['log_interval']
        
        # Model
        self.model = create_cnn_based_model(config).to(device)
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Mixed precision training for RTX 3090
        self.scaler = torch.amp.GradScaler('cuda')
        self.use_amp = True  # Automatic Mixed Precision
        
        # Training state
        self.history = {
            "epoch": [], "train_mse": [], "val_mse": [], "test_mse": [],
            "train_mae_yaw": [], "train_mae_pitch": [], "train_mae_roll": [], "train_mae_overall": [],
            "val_mae_yaw": [], "val_mae_pitch": [], "val_mae_roll": [], "val_mae_overall": [],
            "test_mae_yaw": [], "test_mae_pitch": [], "test_mae_roll": [], "test_mae_overall": []
        }
        
        self.best_val_mae = float('inf')
        self.best_state = None
        self.epochs_no_improve = 0
    
    def load_data(self, splits_path):
        """Load train/val/test data for CNN"""
        # Determine file prefix based on dataset
        prefix = f"{self.dataset}_cnn"
        
        train = np.load(os.path.join(splits_path, f"{prefix}_train.npz"), allow_pickle=True)
        val = np.load(os.path.join(splits_path, f"{prefix}_val.npz"), allow_pickle=True)
        test = np.load(os.path.join(splits_path, f"{prefix}_test.npz"), allow_pickle=True)
        
        # Normalization stats
        self.y_mean = torch.tensor(train["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(train["y_std"], dtype=torch.float32)
        
        # Create datasets
        train_dataset = BIWIImageDataset(train["image_paths"], train["labels"])
        val_dataset = BIWIImageDataset(val["image_paths"], val["labels"])
        test_dataset = BIWIImageDataset(test["image_paths"], test["labels"])
        
        # Create dataloaders - OPTIMIZED FOR RTX 3090
        # Balanced worker count to avoid resource exhaustion
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  # Balanced for stability
            pin_memory=True,
            prefetch_factor=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        
        print(f"{self.dataset.upper()} data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    def mae_per_angle(self, pred_norm, target_norm):
        """Calculate MAE per angle in degrees"""
        pred_deg = pred_norm * self.y_std.to(self.device) + self.y_mean.to(self.device)
        targ_deg = target_norm * self.y_std.to(self.device) + self.y_mean.to(self.device)
        abs_err = (pred_deg - targ_deg).abs()
        mae_each = abs_err.mean(dim=0)
        overall = mae_each.mean()
        return mae_each.detach().cpu(), overall.detach().cpu()
    
    def train_epoch(self):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        for imgs, targets in tqdm(self.train_loader, desc="Training"):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward with automatic mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                preds = self.model(imgs)
                loss = self.criterion(preds, targets)
            
            # Backward with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            all_preds.append(preds.detach())
            all_targets.append(targets.detach())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mae_each, mae_overall = self.mae_per_angle(all_preds, all_targets)
        
        return total_loss / len(self.train_loader), mae_each, mae_overall
    
    def evaluate(self, loader):
        """Evaluate on a dataset with mixed precision"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward with automatic mixed precision
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    preds = self.model(imgs)
                    loss = self.criterion(preds, targets)
                
                total_loss += loss.item()
                all_preds.append(preds)
                all_targets.append(targets)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mae_each, mae_overall = self.mae_per_angle(all_preds, all_targets)
        
        return total_loss / len(loader), mae_each, mae_overall
    
    def train(self):
        """Full training loop with error handling"""
        print(f"\nTraining CNN-based model for {self.epochs} epochs...")
        
        for epoch in range(1, self.epochs + 1):
            try:
                # Train
                train_mse, train_mae_each, train_mae_overall = self.train_epoch()
                
                # Validate
                val_mse, val_mae_each, val_mae_overall = self.evaluate(self.val_loader)
                
                # Test
                test_mse, test_mae_each, test_mae_overall = self.evaluate(self.test_loader)
                
                # Log
                self.history["epoch"].append(epoch)
                self.history["train_mse"].append(train_mse)
                self.history["val_mse"].append(val_mse)
                self.history["test_mse"].append(test_mse)
                
                self.history["train_mae_yaw"].append(train_mae_each[0].item())
                self.history["train_mae_pitch"].append(train_mae_each[1].item())
                self.history["train_mae_roll"].append(train_mae_each[2].item())
                self.history["train_mae_overall"].append(train_mae_overall.item())
                
                self.history["val_mae_yaw"].append(val_mae_each[0].item())
                self.history["val_mae_pitch"].append(val_mae_each[1].item())
                self.history["val_mae_roll"].append(val_mae_each[2].item())
                self.history["val_mae_overall"].append(val_mae_overall.item())
                
                self.history["test_mae_yaw"].append(test_mae_each[0].item())
                self.history["test_mae_pitch"].append(test_mae_each[1].item())
                self.history["test_mae_roll"].append(test_mae_each[2].item())
                self.history["test_mae_overall"].append(test_mae_overall.item())
                
                # Print progress after EVERY epoch
                print(f"\nEpoch {epoch}/{self.epochs}:")
                print(f"  Train MSE={train_mse:.4f} | Val MSE={val_mse:.4f} | Test MSE={test_mse:.4f}")
                print(f"  Train MAE: Yaw={train_mae_each[0]:.2f}° Pitch={train_mae_each[1]:.2f}° Roll={train_mae_each[2]:.2f}° (Avg {train_mae_overall:.2f}°)")
                print(f"  Val   MAE: Yaw={val_mae_each[0]:.2f}° Pitch={val_mae_each[1]:.2f}° Roll={val_mae_each[2]:.2f}° (Avg {val_mae_overall:.2f}°)")
                print(f"  Test  MAE: Yaw={test_mae_each[0]:.2f}° Pitch={test_mae_each[1]:.2f}° Roll={test_mae_each[2]:.2f}° (Avg {test_mae_overall:.2f}°)")
                
                # Early stopping
                if val_mae_overall < self.best_val_mae:
                    self.best_val_mae = val_mae_overall
                    self.best_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'y_mean': self.y_mean,
                        'y_std': self.y_std,
                        'config': self.config
                    }
                    self.epochs_no_improve = 0
                    print(f"  ✓ New best model! (Val MAE: {val_mae_overall:.2f}°)")
                else:
                    self.epochs_no_improve += 1
                    
                    if self.epochs_no_improve >= self.patience:
                        print(f"\n⚠ Early stopping after {epoch} epochs (no improvement for {self.patience} epochs)")
                        break
                        
            except Exception as e:
                print(f"\n❌ Error in epoch {epoch}: {str(e)}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠ Continuing to next epoch...")
                continue
        
        # Load best model
        if self.best_state:
            self.model.load_state_dict(self.best_state['model_state'])
            print(f"\nLoaded best model from epoch {self.best_state['epoch']}")
    
    def save_model(self, save_dir, model_name=None):
        """Save the best model"""
        ensure_dir(save_dir)
        if model_name is None:
            model_name = f"cervical_headpose_cnn_{self.dataset}_best.pth"
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.best_state, save_path)
        return save_path
