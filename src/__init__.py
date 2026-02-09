# src/__init__.py
"""
Head Pose Estimation - Source Package
"""
from .utils import load_config, set_seed, get_device

# Import only modules that exist
try:
    from .cnn_feature_extractor import CNNFeatureExtractor
except ImportError:
    CNNFeatureExtractor = None

try:
    from .data_preprocessing_cnn import BIWIDatasetProcessor
except ImportError:
    BIWIDatasetProcessor = None

try:
    from .data_preprocessing_300wlp import Dataset300WLPProcessor
except ImportError:
    Dataset300WLPProcessor = None

try:
    from .trainer_cnn import CNNHeadPoseTrainer
except ImportError:
    CNNHeadPoseTrainer = None

__version__ = "1.0.0"

__all__ = [
    'load_config',
    'set_seed',
    'get_device',
    'CNNFeatureExtractor',
    'BIWIDatasetProcessor',
    'Dataset300WLPProcessor',
    'CNNHeadPoseTrainer',
]