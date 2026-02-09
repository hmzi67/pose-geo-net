"""
CNN-based Feature Extraction with Multiple Backbone Support
Supports: ResNet18/34/50, MobileNetV3, EfficientNet
This replaces MediaPipe with a learnable CNN feature extractor
"""
import torch
import torch.nn as nn
import torchvision.models as models


class FlexibleBackboneFeatureExtractor(nn.Module):
    """
    Flexible backbone feature extractor supporting multiple architectures.
    Extracts learned features from raw images, replacing MediaPipe landmarks.
    
    Supported Backbones:
    - ResNet18 (512 features)
    - ResNet34 (512 features)
    - ResNet50 (2048 features)
    - MobileNetV3-Large (960 features)
    - MobileNetV3-Small (576 features)
    - EfficientNet-B0 (1280 features)
    
    Architecture:
    - Uses pretrained backbone
    - Splits features into face-focused and pose-focused branches
    - Outputs separate face and pose feature vectors
    """
    
    BACKBONE_DIMS = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'mobilenet_v3_large': 960,
        'mobilenet_v3_small': 576,
        'efficientnet_b0': 1280
    }
    
    def __init__(self, 
                 backbone='resnet18',
                 face_feature_dim=1404,
                 pose_feature_dim=99,
                 pretrained=True,
                 freeze_backbone=False):
        """
        Args:
            backbone: Backbone architecture name
            face_feature_dim: Output dimension for face features (default 1404 to match MediaPipe)
            pose_feature_dim: Output dimension for pose features (default 99 to match MediaPipe)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone (only train feature heads)
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load appropriate backbone
        if backbone not in self.BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(self.BACKBONE_DIMS.keys())}")
        
        backbone_dim = self.BACKBONE_DIMS[backbone]
        
        # Create backbone
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        elif backbone == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Shared feature processing (adaptive to backbone dimension)
        self.shared_fc = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3)
        )
        
        # Face feature head (mimics face landmark features)
        self.face_head = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, face_feature_dim)
        )
        
        # Pose feature head (mimics pose landmark features)
        self.pose_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, pose_feature_dim)
        )
        
        self.face_feature_dim = face_feature_dim
        self.pose_feature_dim = pose_feature_dim
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [batch, 3, H, W]
        
        Returns:
            face_features: [batch, face_feature_dim]
            pose_features: [batch, pose_feature_dim]
        """
        # Extract backbone features
        features = self.backbone(x)  # [batch, backbone_dim, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, backbone_dim]
        
        # Shared processing
        shared = self.shared_fc(features)  # [batch, 1024]
        
        # Split into face and pose features
        face_features = self.face_head(shared)  # [batch, face_feature_dim]
        pose_features = self.pose_head(shared)  # [batch, pose_feature_dim]
        
        return face_features, pose_features


class CNNBasedHeadPoseModel(nn.Module):
    """
    Complete end-to-end model: CNN Feature Extractor + Dual Branch Architecture
    """
    
    def __init__(self,
                 backbone='resnet18',
                 face_feature_dim=1404,
                 pose_feature_dim=99,
                 face_encoder_hidden=[512, 256],
                 pose_encoder_hidden=[128, 64],
                 fusion_dim=256,
                 regression_hidden=128,
                 output_dim=3,
                 pretrained_cnn=True,
                 freeze_backbone=False):
        """
        Args:
            backbone: CNN backbone architecture name
            face_feature_dim: Face feature dimension from CNN
            pose_feature_dim: Pose feature dimension from CNN
            face_encoder_hidden: Hidden layers for face encoder
            pose_encoder_hidden: Hidden layers for pose encoder
            fusion_dim: Fusion layer dimension
            regression_hidden: Regression head hidden dimension
            output_dim: Output dimension (3 for yaw, pitch, roll)
            pretrained_cnn: Use pretrained backbone
            freeze_backbone: Freeze CNN backbone
        """
        super().__init__()
        
        # Import dual-branch components
        from models.mlp_model import FaceEncoder, PoseEncoder, FeatureFusion, RegressionHead
        
        # CNN Feature Extractor (flexible backbone)
        self.feature_extractor = FlexibleBackboneFeatureExtractor(
            backbone=backbone,
            face_feature_dim=face_feature_dim,
            pose_feature_dim=pose_feature_dim,
            pretrained=pretrained_cnn,
            freeze_backbone=freeze_backbone
        )
        
        # Dual-branch architecture (same as before)
        self.face_encoder = FaceEncoder(
            input_dim=face_feature_dim,
            hidden_dims=face_encoder_hidden,
            dropout_rate=0.1
        )
        
        self.pose_encoder = PoseEncoder(
            input_dim=pose_feature_dim,
            hidden_dims=pose_encoder_hidden,
            dropout_rate=0.1
        )
        
        self.fusion = FeatureFusion(
            face_dim=self.face_encoder.output_dim,
            pose_dim=self.pose_encoder.output_dim,
            fusion_dim=fusion_dim,
            dropout_rate=0.1
        )
        
        self.regression_head = RegressionHead(
            input_dim=fusion_dim,
            hidden_dim=regression_hidden,
            output_dim=output_dim,
            dropout_rate=0.05
        )
    
    def forward(self, x):
        """
        End-to-end forward pass
        
        Args:
            x: Input images [batch, 3, H, W]
        
        Returns:
            angles: Predicted angles [batch, 3] (yaw, pitch, roll)
        """
        # Extract features from CNN
        face_features, pose_features = self.feature_extractor(x)
        
        # Encode separately
        face_encoded = self.face_encoder(face_features)
        pose_encoded = self.pose_encoder(pose_features)
        
        # Fuse
        fused = self.fusion(face_encoded, pose_encoded)
        
        # Predict angles
        angles = self.regression_head(fused)
        
        return angles


def create_cnn_based_model(config, backbone=None):
    """
    Create CNN-based model from config
    
    Args:
        config: Configuration dictionary
        backbone: Optional backbone architecture to override config (resnet18, resnet34, resnet50, mobilenet_v3_large, efficientnet_b0)
    """
    model_cfg = config['model']
    
    # Use provided backbone or fall back to config
    backbone_name = backbone if backbone is not None else model_cfg.get('backbone', 'resnet18')
    
    return CNNBasedHeadPoseModel(
        backbone=backbone_name,
        face_feature_dim=model_cfg.get('face_input_dim', 1404),
        pose_feature_dim=model_cfg.get('pose_input_dim', 99),
        face_encoder_hidden=model_cfg.get('face_hidden', [512, 256]),
        pose_encoder_hidden=model_cfg.get('pose_hidden', [128, 64]),
        fusion_dim=model_cfg.get('fusion_dim', 256),
        regression_hidden=model_cfg.get('regression_hidden', 128),
        output_dim=model_cfg['output_dim'],
        pretrained_cnn=model_cfg.get('pretrained_cnn', True),
        freeze_backbone=model_cfg.get('freeze_cnn_backbone', False)
    )
