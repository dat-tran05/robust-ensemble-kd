"""
ResNet models with feature extraction hooks for knowledge distillation.
Supports extracting intermediate features (layer1-4) and penultimate features.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetWithFeatures(nn.Module):
    """
    ResNet wrapper that extracts intermediate features for distillation.
    
    Usage:
        model = ResNetWithFeatures('resnet50', num_classes=2, pretrained=True)
        logits, features = model(x, return_features=True)
        # features['layer4'] contains penultimate features
    """
    
    def __init__(self, arch='resnet50', num_classes=2, pretrained=True):
        super().__init__()
        
        # Load base model
        if arch == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 2048
        elif arch == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Split into feature extractor and classifier
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.avgpool = base.avgpool
        
        # New classifier head
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        self.arch = arch
        self.num_classes = num_classes
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input images [B, 3, 224, 224]
            return_features: If True, return (logits, feature_dict)
            
        Returns:
            logits: Class logits [B, num_classes]
            features (optional): Dict with layer1-4 and pooled features
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        # Pool and classify
        pooled = self.avgpool(l4)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        
        if return_features:
            features = {
                'layer1': l1,
                'layer2': l2,
                'layer3': l3,
                'layer4': l4,      # Penultimate (before avgpool)
                'pooled': pooled,  # After avgpool, before fc
            }
            return logits, features
        
        return logits
    
    def get_penultimate_features(self, x):
        """Convenience method to get pooled penultimate features."""
        _, features = self.forward(x, return_features=True)
        return features['pooled']


class FeatureAdapter(nn.Module):
    """
    Adapter to match student feature dimensions to teacher dimensions.
    Uses 1x1 convolution for spatial features or linear for pooled features.
    
    Args:
        student_dim: Student feature dimension
        teacher_dim: Teacher feature dimension
        spatial: If True, use Conv2d; if False, use Linear
    """
    
    def __init__(self, student_dim, teacher_dim, spatial=True):
        super().__init__()
        
        if student_dim == teacher_dim:
            self.adapter = nn.Identity()
        elif spatial:
            self.adapter = nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False)
        else:
            self.adapter = nn.Linear(student_dim, teacher_dim, bias=False)
    
    def forward(self, x):
        return self.adapter(x)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_teacher_model(arch='resnet50', num_classes=2, pretrained=True):
    """Create a teacher model (typically ResNet-50)."""
    return ResNetWithFeatures(arch, num_classes, pretrained)


def get_student_model(arch='resnet18', num_classes=2, pretrained=True):
    """Create a student model (typically ResNet-18, smaller)."""
    return ResNetWithFeatures(arch, num_classes, pretrained)


def create_feature_adapter(student_arch='resnet18', teacher_arch='resnet50', layer='pooled'):
    """
    Create adapter for feature distillation between student and teacher.
    
    Common configurations:
        - ResNet-18 -> ResNet-50, pooled: 512 -> 2048
        - ResNet-18 -> ResNet-50, layer4: 512 -> 2048 (spatial)
    """
    # Feature dimensions by architecture and layer
    dims = {
        'resnet18': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'pooled': 512},
        'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'pooled': 2048},
    }
    
    student_dim = dims[student_arch][layer]
    teacher_dim = dims[teacher_arch][layer]
    spatial = (layer != 'pooled')
    
    return FeatureAdapter(student_dim, teacher_dim, spatial=spatial)


# =============================================================================
# UTILITY: Load checkpoint
# =============================================================================

def load_checkpoint(model, checkpoint_path, strict=True):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Return metadata if available
    return {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'state_dict']}


def load_teacher_checkpoint(model, checkpoint_path, strict=False):
    """
    Load teacher checkpoint (works for both ERM and debiased formats).

    Handles various checkpoint formats:
    - ERM checkpoints from DFR repo: {'model': state_dict}
    - Debiased checkpoints we create: {'model_state_dict': state_dict}
    - DataParallel 'module.' prefix
    - Various backbone prefixes

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: If False, allows missing/extra keys (default: False)

    Returns:
        metadata: Dict with non-weight checkpoint contents
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict from various formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix from DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Handle potential backbone prefix
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

    # Handle model.X prefix (some repos wrap the model)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    # Try to match our model's key structure
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Check for common prefix issues
    if len(model_keys & ckpt_keys) == 0:
        # Keys don't match at all - try to find a mapping
        sample_model_key = list(model_keys)[0]
        sample_ckpt_key = list(ckpt_keys)[0]
        print(f"Warning: No matching keys. Model: {sample_model_key}, Ckpt: {sample_ckpt_key}")

    # Load with flexibility
    result = model.load_state_dict(state_dict, strict=strict)

    missing = result.missing_keys if hasattr(result, 'missing_keys') else []
    unexpected = result.unexpected_keys if hasattr(result, 'unexpected_keys') else []

    print(f"Loaded checkpoint from {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)} (e.g., {missing[:3]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")
    if not missing and not unexpected:
        print(f"  All keys matched!")

    # Return metadata
    if isinstance(checkpoint, dict):
        return {k: v for k, v in checkpoint.items()
                if k not in ['model', 'state_dict', 'model_state_dict']}
    return {}


def load_teachers_from_dir(checkpoint_dir, model_fn, num_teachers=None, device='cuda'):
    """
    Load multiple teacher models from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_fn: Function to create model (e.g., lambda: get_teacher_model('resnet50', 2))
        num_teachers: Number of teachers to load (None = all found)
        device: Device to load models onto

    Returns:
        teachers: List of loaded teacher models in eval mode
    """
    import os

    # Find checkpoint files
    ckpt_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pt') or f.endswith('.pth')
    ])

    if num_teachers is not None:
        ckpt_files = ckpt_files[:num_teachers]

    print(f"Found {len(ckpt_files)} checkpoints in {checkpoint_dir}")

    teachers = []
    for ckpt_file in ckpt_files:
        model = model_fn()
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        load_teacher_checkpoint(model, ckpt_path)
        model = model.to(device)
        model.eval()
        teachers.append(model)
        print(f"  Loaded: {ckpt_file}")

    return teachers


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing models...")
    
    # Test teacher (ResNet-50)
    teacher = get_teacher_model('resnet50', num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    
    logits, features = teacher(x, return_features=True)
    print(f"\nTeacher (ResNet-50):")
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Features:")
    for name, feat in features.items():
        print(f"    {name}: {feat.shape}")
    
    # Test student (ResNet-18)
    student = get_student_model('resnet18', num_classes=2)
    s_logits, s_features = student(x, return_features=True)
    print(f"\nStudent (ResNet-18):")
    print(f"  Logits: {s_logits.shape}")
    print(f"  Features:")
    for name, feat in s_features.items():
        print(f"    {name}: {feat.shape}")
    
    # Test adapter
    adapter = create_feature_adapter('resnet18', 'resnet50', 'pooled')
    adapted = adapter(s_features['pooled'])
    print(f"\nAdapter (pooled):")
    print(f"  Student pooled: {s_features['pooled'].shape}")
    print(f"  Adapted: {adapted.shape}")
    print(f"  Teacher pooled: {features['pooled'].shape}")
