"""
Fusion Strategy Ablation Variants for Head Pose Estimation

This module implements different fusion strategies to compare:
1. Early Fusion: Concatenate CNN features before any branch-specific processing
2. Attention Fusion: Use attention mechanism to weight face/pose importance
3. Bilinear Pooling: Compute outer product for richer feature interactions
4. Late Fusion (Baseline): Separate encoding then concatenation (current approach)

Purpose: Ablation study to justify the late fusion design choice (Table 4 in paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EarlyFusionModel(nn.Module):
    """
    Early Fusion: CNN → Concatenate all features → FC layers → Output
    
    This is the simplest baseline where features are combined immediately
    after CNN extraction, before any modality-specific processing.
    Expected to perform worst due to premature integration preventing
    modality-specific learning.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 960
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Early fusion: Simple FC head directly on backbone features
        # No separate branches - just concatenate everything and process
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """Forward pass - early fusion (no branch split)"""
        features = self.backbone(x)
        output = self.fc_head(features)
        return output


class AttentionFusionModel(nn.Module):
    """
    Attention-based Fusion: CNN → Separate heads → Attention-weighted combination → Output
    
    Uses a learned attention mechanism to dynamically weight the importance
    of face and pose features. Should perform better than early/bilinear 
    but worse than late fusion with learned weights.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, 
                 face_dim=256, pose_dim=64, output_dim=3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 960
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.face_dim = face_dim
        self.pose_dim = pose_dim
        
        # Shared feature processing (geometry-guided approach)
        self.shared_fc = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3)
        )
        shared_dim = 1024
        
        # Separate feature heads (project shared features to face/pose dimensions)
        self.face_head = nn.Sequential(
            nn.Linear(shared_dim, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, face_dim),
            nn.ReLU(),
            nn.BatchNorm1d(face_dim)
        )
        
        self.pose_head = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, pose_dim),
            nn.ReLU(),
            nn.BatchNorm1d(pose_dim)
        )
        
        # Attention mechanism
        # Computes attention weights for face and pose features
        combined_dim = face_dim + pose_dim
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2),  # 2 weights: one for face, one for pose
            nn.Softmax(dim=1)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """Forward pass with attention-based fusion"""
        # Extract backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Shared feature processing
        shared = self.shared_fc(features)  # [batch, 1024]
        
        # Separate heads
        face_feat = self.face_head(shared)  # [batch, face_dim]
        pose_feat = self.pose_head(shared)  # [batch, pose_dim]
        
        # Concatenate for attention computation
        combined = torch.cat([face_feat, pose_feat], dim=1)  # [batch, face_dim + pose_dim]
        
        # Compute attention weights
        attn_weights = self.attention(combined)  # [batch, 2]
        
        # Apply attention-weighted fusion
        # Scale face and pose features by their attention weights
        face_weight = attn_weights[:, 0:1]  # [batch, 1]
        pose_weight = attn_weights[:, 1:2]  # [batch, 1]
        
        weighted_face = face_feat * face_weight  # [batch, face_dim]
        weighted_pose = pose_feat * pose_weight  # [batch, pose_dim]
        
        # Concatenate weighted features
        fused = torch.cat([weighted_face, weighted_pose], dim=1)  # [batch, combined_dim]
        
        # Predict angles
        output = self.regression_head(fused)
        return output


class BilinearPoolingModel(nn.Module):
    """
    Bilinear Pooling Fusion: CNN → Separate heads → Outer product → Output
    
    Computes the outer product of face and pose features to capture
    second-order interactions. More expressive than concatenation but
    increases dimensionality significantly.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True,
                 face_dim=64, pose_dim=32, output_dim=3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 960
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.face_dim = face_dim
        self.pose_dim = pose_dim
        
        # Separate feature heads (smaller dimensions for bilinear to manage size)
        self.face_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, face_dim),
            nn.ReLU()
        )
        
        self.pose_head = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, pose_dim),
            nn.ReLU()
        )
        
        # Bilinear pooling output dimension: face_dim * pose_dim
        bilinear_dim = face_dim * pose_dim
        
        # Dimensionality reduction after bilinear pooling
        self.bilinear_reduction = nn.Sequential(
            nn.Linear(bilinear_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """Forward pass with bilinear pooling fusion"""
        # Extract backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Separate heads
        face_feat = self.face_head(features)  # [batch, face_dim]
        pose_feat = self.pose_head(features)  # [batch, pose_dim]
        
        # Bilinear pooling: outer product
        # [batch, face_dim, 1] x [batch, 1, pose_dim] -> [batch, face_dim, pose_dim]
        bilinear = torch.bmm(
            face_feat.unsqueeze(2),  # [batch, face_dim, 1]
            pose_feat.unsqueeze(1)   # [batch, 1, pose_dim]
        )
        
        # Flatten bilinear features
        bilinear = bilinear.view(bilinear.size(0), -1)  # [batch, face_dim * pose_dim]
        
        # Signed square root and L2 normalization (standard bilinear pooling)
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-8)
        bilinear = F.normalize(bilinear, p=2, dim=1)
        
        # Reduce dimension
        fused = self.bilinear_reduction(bilinear)
        
        # Predict angles
        output = self.regression_head(fused)
        return output


class LateFusionModel(nn.Module):
    """
    Late Fusion with Learned Weights (Our Approach)
    
    CNN → Separate Face/Pose Encoding → Concatenation → Fusion FC → Output
    
    This is the baseline dual-branch architecture. Features are encoded 
    separately to preserve modality-specific characteristics, then fused
    through learned weights.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True,
                 face_feature_dim=1404, pose_feature_dim=99,
                 face_hidden=[512, 256], pose_hidden=[128, 64],
                 fusion_dim=256, regression_hidden=128, output_dim=3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            backbone_dim = 2048
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 960
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = model.features
            self.backbone.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3)
        )
        
        # Face feature head
        self.face_head = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, face_feature_dim)
        )
        
        # Pose feature head
        self.pose_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, pose_feature_dim)
        )
        
        # Face encoder (deeper processing)
        face_layers = []
        prev_dim = face_feature_dim
        for hidden_dim in face_hidden:
            face_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.face_encoder = nn.Sequential(*face_layers)
        face_output_dim = face_hidden[-1]
        
        # Pose encoder (deeper processing)
        pose_layers = []
        prev_dim = pose_feature_dim
        for hidden_dim in pose_hidden:
            pose_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.pose_encoder = nn.Sequential(*pose_layers)
        pose_output_dim = pose_hidden[-1]
        
        # Late fusion layer with learned weights
        self.fusion = nn.Sequential(
            nn.Linear(face_output_dim + pose_output_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.1)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, regression_hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(regression_hidden, output_dim)
        )
    
    def forward(self, x):
        """Forward pass with late fusion"""
        # Extract backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Shared processing
        shared = self.shared_fc(features)
        
        # Separate heads
        face_feat = self.face_head(shared)
        pose_feat = self.pose_head(shared)
        
        # Deep encoding (separate paths)
        face_encoded = self.face_encoder(face_feat)
        pose_encoded = self.pose_encoder(pose_feat)
        
        # Late fusion
        combined = torch.cat([face_encoded, pose_encoded], dim=1)
        fused = self.fusion(combined)
        
        # Predict angles
        output = self.regression_head(fused)
        return output


def create_fusion_variant(fusion_type, backbone='resnet18', pretrained=True, output_dim=3, 
                         face_dim=256, pose_dim=64):
    """
    Factory function to create fusion strategy variants.
    
    Args:
        fusion_type: 'early', 'attention', 'bilinear', or 'late'
        backbone: Backbone architecture name
        pretrained: Use pretrained weights
        output_dim: Output dimension (3 for yaw, pitch, roll)
        face_dim: Face feature dimension (default 256, use 1404 for geometry-guided)
        pose_dim: Pose feature dimension (default 64, use 99 for geometry-guided)
    
    Returns:
        Model instance
    """
    if fusion_type == 'early':
        return EarlyFusionModel(
            backbone=backbone,
            pretrained=pretrained,
            output_dim=output_dim
        )
    
    elif fusion_type == 'attention':
        return AttentionFusionModel(
            backbone=backbone,
            pretrained=pretrained,
            face_dim=face_dim,
            pose_dim=pose_dim,
            output_dim=output_dim
        )
    
    elif fusion_type == 'bilinear':
        return BilinearPoolingModel(
            backbone=backbone,
            pretrained=pretrained,
            face_dim=min(face_dim, 64),  # Smaller for bilinear to avoid explosion
            pose_dim=min(pose_dim, 32),
            output_dim=output_dim
        )
    
    elif fusion_type == 'late':
        return LateFusionModel(
            backbone=backbone,
            pretrained=pretrained,
            output_dim=output_dim
        )
    
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}. "
                        f"Choose from: early, attention, bilinear, late")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test fusion variants"""
    print("=" * 70)
    print("Testing Fusion Strategy Variants")
    print("=" * 70)
    
    variants = ['early', 'attention', 'bilinear', 'late']
    
    for variant in variants:
        print(f"\n{variant.upper()} FUSION")
        print("-" * 70)
        
        model = create_fusion_variant(variant, backbone='resnet18')
        params = count_parameters(model)
        
        print(f"  Total Parameters: {params:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  ✓ Forward pass successful")
    
    print("\n" + "=" * 70)
    print("All fusion variants working correctly!")
    print("=" * 70)
