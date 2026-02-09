"""
Architecture Ablation Variants for Head Pose Estimation

This module implements different architectural approaches to compare:
1. Single-branch Direct: CNN → FC layers (minimal processing)
2. Single-branch MLP: CNN → MLP → FC (more processing but no split)
3. Dual-branch (Original): CNN → Face/Pose branches → Fusion

Purpose: Justify the dual-branch design choice in ablation studies.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SingleBranchDirect(nn.Module):
    """
    Single-branch architecture: Direct mapping from CNN to pose angles.
    CNN Backbone → Flatten → FC → Output
    
    This is the simplest baseline: no feature splitting, minimal processing.
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
        
        # Simple direct regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.regression_head(features)
        return output


class SingleBranchMLP(nn.Module):
    """
    Single-branch with MLP: More processing but no feature splitting.
    CNN Backbone → MLP (multiple layers) → Output
    
    This tests whether adding more layers helps without splitting into branches.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, 
                 mlp_hidden=[512, 256, 128], output_dim=3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone (same as SingleBranchDirect)
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
        
        # Build MLP
        layers = [nn.Flatten()]
        prev_dim = backbone_dim
        
        for i, hidden_dim in enumerate(mlp_hidden):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout (decreasing)
            if i == 0:
                layers.append(nn.Dropout(0.3))
            elif i == 1:
                layers.append(nn.Dropout(0.2))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.mlp(features)
        return output


def create_architecture_variant(variant_type, backbone='resnet18', 
                                 pretrained=True, output_dim=3, **kwargs):
    """
    Factory function to create architecture variants.
    
    Args:
        variant_type: 'single_direct', 'single_mlp', or 'dual_branch'
        backbone: Backbone architecture name
        pretrained: Use pretrained weights
        output_dim: Output dimension (3 for yaw, pitch, roll)
        **kwargs: Additional arguments for specific variants
    
    Returns:
        Model instance
    """
    if variant_type == 'single_direct':
        return SingleBranchDirect(
            backbone=backbone,
            pretrained=pretrained,
            output_dim=output_dim
        )
    
    elif variant_type == 'single_mlp':
        mlp_hidden = kwargs.get('mlp_hidden', [512, 256, 128])
        return SingleBranchMLP(
            backbone=backbone,
            pretrained=pretrained,
            mlp_hidden=mlp_hidden,
            output_dim=output_dim
        )
    
    elif variant_type == 'dual_branch':
        # Import here to avoid circular dependency
        import sys
        sys.path.append('.')
        from src.cnn_feature_extractor import create_cnn_based_model
        from src.utils import load_config
        
        config = load_config()
        return create_cnn_based_model(config, backbone=backbone)
    
    else:
        raise ValueError(f"Unknown variant_type: {variant_type}. "
                        f"Choose from: single_direct, single_mlp, dual_branch")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test architecture variants"""
    print("=" * 70)
    print("Testing Architecture Variants")
    print("=" * 70)
    
    variants = ['single_direct', 'single_mlp', 'dual_branch']
    
    for variant in variants:
        print(f"\n{variant.upper()}")
        print("-" * 70)
        
        model = create_architecture_variant(variant, backbone='resnet18')
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
    print("All variants working correctly!")
    print("=" * 70)
