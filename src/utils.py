"""
Utility functions for head pose estimation project
"""
import os
import yaml
import json
import numpy as np
import torch
import random
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_pose_txt(pose_path):
    """
    Parse BIWI pose annotation file
    Returns rotation matrix R and translation vector t
    """
    with open(pose_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    # First 3 lines: rotation matrix
    R = np.array([
        [float(x) for x in lines[i].split()[:3]] 
        for i in range(3)
    ], dtype=np.float32)
    
    # 4th line: translation (optional)
    tvals = [float(x) for x in lines[3].split()] if len(lines) > 3 else [0, 0, 0]
    t = np.array(tvals, dtype=np.float32)
    
    return R, t


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll) in degrees
    Fixed version with correct yaw/pitch assignment
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        # Corrected: pitch is up/down, yaw is left/right
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    
    return np.degrees([yaw, pitch, roll]).astype(np.float32)


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_json(data, filepath):
    """Save data to JSON file with numpy type conversion"""
    data_serializable = convert_to_serializable(data)
    with open(filepath, 'w') as f:
        json.dump(data_serializable, f, indent=2)


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_device(config):
    """Get torch device based on config and availability"""
    device_name = config.get('device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')