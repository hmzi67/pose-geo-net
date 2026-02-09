"""
MLP Model for Head Pose Estimation
"""
import torch
import torch.nn as nn


class HeadPoseMLP(nn.Module):
    """Multi-layer Perceptron for head pose regression"""
    
    def __init__(self, input_dim=1503, output_dim=3, 
                 hidden_layers=[768, 256, 128], 
                 dropout_rates=[0.15, 0.10, 0.0]):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (3 for yaw, pitch, roll)
            hidden_layers: List of hidden layer sizes
            dropout_rates: List of dropout rates for each hidden layer
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.net(x)


class FaceEncoder(nn.Module):
    """Encodes facial landmarks while preserving geometric characteristics"""
    
    def __init__(self, input_dim=1404, hidden_dims=[512, 256], dropout_rate=0.1):
        """
        Args:
            input_dim: Facial landmark dimension (468 landmarks * 3 = 1404)
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        """Encode face landmarks"""
        return self.encoder(x)


class PoseEncoder(nn.Module):
    """Encodes upper-body pose landmarks"""
    
    def __init__(self, input_dim=99, hidden_dims=[128, 64], dropout_rate=0.1):
        """
        Args:
            input_dim: Pose landmark dimension (33 landmarks * 3 = 99)
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        """Encode pose landmarks"""
        return self.encoder(x)


class FeatureFusion(nn.Module):
    """Fuses face and pose features for joint reasoning"""
    
    def __init__(self, face_dim=256, pose_dim=64, fusion_dim=256, dropout_rate=0.1):
        """
        Args:
            face_dim: Face encoder output dimension
            pose_dim: Pose encoder output dimension
            fusion_dim: Fusion layer output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(face_dim + pose_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout_rate)
        )
        self.output_dim = fusion_dim
    
    def forward(self, face_feat, pose_feat):
        """Fuse face and pose features"""
        combined = torch.cat([face_feat, pose_feat], dim=1)
        return self.fusion(combined)


class RegressionHead(nn.Module):
    """Lightweight regression head for angle prediction"""
    
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=3, dropout_rate=0.05):
        """
        Args:
            input_dim: Input feature dimension from fusion
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (3 for yaw, pitch, roll)
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Predict angles"""
        return self.regressor(x)


class HeadPoseDualBranchMLP(nn.Module):
    """
    Landmark-aware dual-branch regression architecture for 3D head pose estimation.
    
    Architecture:
    1. FaceEncoder: Encodes 468 facial landmarks (1404 dims)
    2. PoseEncoder: Encodes 33 upper-body pose landmarks (99 dims)
    3. FeatureFusion: Fuses face and pose representations
    4. RegressionHead: Maps fused features to yaw, pitch, roll angles
    
    This design introduces structured inductive bias by processing face and pose
    landmarks separately, preserving their geometric characteristics before fusion.
    """
    
    def __init__(self, 
                 face_input_dim=1404,
                 pose_input_dim=99,
                 face_hidden=[512, 256],
                 pose_hidden=[128, 64],
                 fusion_dim=256,
                 regression_hidden=128,
                 output_dim=3,
                 dropout_rates={'face': 0.1, 'pose': 0.1, 'fusion': 0.1, 'regression': 0.05}):
        """
        Args:
            face_input_dim: Facial landmark dimension (468 * 3 = 1404)
            pose_input_dim: Pose landmark dimension (33 * 3 = 99)
            face_hidden: Hidden dimensions for face encoder
            pose_hidden: Hidden dimensions for pose encoder
            fusion_dim: Fusion layer dimension
            regression_hidden: Hidden dimension for regression head
            output_dim: Output dimension (3 for yaw, pitch, roll)
            dropout_rates: Dict with dropout rates for each component
        """
        super().__init__()
        
        # Separate encoders
        self.face_encoder = FaceEncoder(
            input_dim=face_input_dim,
            hidden_dims=face_hidden,
            dropout_rate=dropout_rates['face']
        )
        
        self.pose_encoder = PoseEncoder(
            input_dim=pose_input_dim,
            hidden_dims=pose_hidden,
            dropout_rate=dropout_rates['pose']
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(
            face_dim=self.face_encoder.output_dim,
            pose_dim=self.pose_encoder.output_dim,
            fusion_dim=fusion_dim,
            dropout_rate=dropout_rates['fusion']
        )
        
        # Regression head
        self.regression_head = RegressionHead(
            input_dim=fusion_dim,
            hidden_dim=regression_hidden,
            output_dim=output_dim,
            dropout_rate=dropout_rates['regression']
        )
        
        # Store dimensions for checkpoint saving
        self.face_input_dim = face_input_dim
        self.pose_input_dim = pose_input_dim
    
    def forward(self, x):
        """
        Forward pass with automatic feature splitting
        
        Args:
            x: Input features, either:
               - Concatenated [batch, 1503] (face + pose)
               - Tuple of (face_features, pose_features)
        
        Returns:
            Predicted angles [batch, 3] (yaw, pitch, roll)
        """
        # Handle both single tensor and tuple inputs
        if isinstance(x, tuple):
            face_feat_input, pose_feat_input = x
        else:
            # Split concatenated features
            face_feat_input = x[:, :self.face_input_dim]
            pose_feat_input = x[:, self.face_input_dim:self.face_input_dim + self.pose_input_dim]
        
        # Encode separately
        face_encoded = self.face_encoder(face_feat_input)
        pose_encoded = self.pose_encoder(pose_feat_input)
        
        # Fuse features
        fused = self.fusion(face_encoded, pose_encoded)
        
        # Predict angles
        angles = self.regression_head(fused)
        
        return angles


def create_model(config):
    """Create model from config"""
    model_cfg = config['model']
    
    # Check if dual-branch architecture is requested
    if 'architecture' in model_cfg and model_cfg['architecture'] == 'dual_branch':
        # Create dual-branch model
        dropout_rates = model_cfg.get('dropout_rates_dual', {
            'face': 0.1, 'pose': 0.1, 'fusion': 0.1, 'regression': 0.05
        })
        
        return HeadPoseDualBranchMLP(
            face_input_dim=model_cfg.get('face_input_dim', 1404),
            pose_input_dim=model_cfg.get('pose_input_dim', 99),
            face_hidden=model_cfg.get('face_hidden', [512, 256]),
            pose_hidden=model_cfg.get('pose_hidden', [128, 64]),
            fusion_dim=model_cfg.get('fusion_dim', 256),
            regression_hidden=model_cfg.get('regression_hidden', 128),
            output_dim=model_cfg['output_dim'],
            dropout_rates=dropout_rates
        )
    else:
        # Create original single-branch model (backward compatibility)
        return HeadPoseMLP(
            input_dim=model_cfg['input_dim'],
            output_dim=model_cfg['output_dim'],
            hidden_layers=model_cfg['hidden_layers'],
            dropout_rates=model_cfg['dropout_rates']
        )


def load_model_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict with normalization stats
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model type from checkpoint
    if 'architecture' in checkpoint and checkpoint['architecture'] == 'dual_branch':
        # Load dual-branch model
        model = HeadPoseDualBranchMLP(
            face_input_dim=checkpoint.get('face_input_dim', 1404),
            pose_input_dim=checkpoint.get('pose_input_dim', 99),
            face_hidden=checkpoint.get('face_hidden', [512, 256]),
            pose_hidden=checkpoint.get('pose_hidden', [128, 64]),
            fusion_dim=checkpoint.get('fusion_dim', 256),
            regression_hidden=checkpoint.get('regression_hidden', 128),
            output_dim=checkpoint['output_dim']
        )
    else:
        # Load original model (backward compatibility)
        model = HeadPoseMLP(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim']
        )
    
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def save_model_checkpoint(model, save_path, **kwargs):
    """
    Save model checkpoint with additional metadata
    
    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        **kwargs: Additional data to save (history, normalization stats, etc.)
    """
    checkpoint = {
        'model_state': model.state_dict(),
        **kwargs
    }
    
    # Add architecture-specific metadata
    if isinstance(model, HeadPoseDualBranchMLP):
        checkpoint['architecture'] = 'dual_branch'
        checkpoint['face_input_dim'] = model.face_input_dim
        checkpoint['pose_input_dim'] = model.pose_input_dim
        checkpoint['output_dim'] = model.regression_head.regressor[-1].out_features
        # Save architecture details
        checkpoint['face_hidden'] = [layer.out_features for layer in model.face_encoder.encoder if isinstance(layer, nn.Linear)]
        checkpoint['pose_hidden'] = [layer.out_features for layer in model.pose_encoder.encoder if isinstance(layer, nn.Linear)]
        checkpoint['fusion_dim'] = model.fusion.output_dim
        checkpoint['regression_hidden'] = model.regression_head.regressor[0].out_features
    else:
        checkpoint['architecture'] = 'single_branch'
        checkpoint['input_dim'] = model.net[0].in_features
        checkpoint['output_dim'] = model.net[-1].out_features
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")
