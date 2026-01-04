"""
Configuration for experiments.
All hyperparameters in one place for easy modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Base configuration."""
    
    # Data
    data_dir: str = './data/waterbirds_v1.0'
    batch_size: int = 128
    num_workers: int = 2
    augment: bool = True
    
    # Model
    teacher_arch: str = 'resnet50'
    student_arch: str = 'resnet18'
    num_classes: int = 2
    pretrained: bool = True  # Use ImageNet pretrained weights
    
    # Training
    epochs: int = 50
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = 'cosine'  # 'cosine' or 'step'
    
    # Distillation
    temperature: float = 4.0
    alpha: float = 1.0  # 1.0 = pure KD, <1 adds class labels
    gamma: float = 0.0  # >0 adds feature distillation
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_freq: int = 10  # Save every N epochs
    
    # Misc
    seed: int = 42
    device: str = 'cuda'


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# Teacher training (ERM)
TEACHER_CONFIG = Config(
    epochs=100,
    lr=0.001,
    weight_decay=1e-3,
    batch_size=32,
)

# Baseline: AGRE-KD style (pure KD, no class labels, no features)
BASELINE_CONFIG = Config(
    alpha=1.0,
    gamma=0.0,
    epochs=30,
    lr=0.001,
)

# Experiment 1: Add class labels (α < 1)
EXP1_CONFIGS = {
    'exp1_alpha05': Config(alpha=0.5, gamma=0.0, epochs=30),
    'exp1_alpha07': Config(alpha=0.7, gamma=0.0, epochs=30),
    'exp1_alpha09': Config(alpha=0.9, gamma=0.0, epochs=30),
}

# Experiment 2: Add feature distillation (γ > 0)
EXP2_CONFIGS = {
    'exp2_gamma01': Config(alpha=1.0, gamma=0.1, epochs=30),
    'exp2_gamma025': Config(alpha=1.0, gamma=0.25, epochs=30),
}

# Experiment 3: Combined (α < 1, γ > 0)
EXP3_CONFIGS = {
    'exp3_a07_g01': Config(alpha=0.7, gamma=0.1, epochs=30),
    'exp3_a07_g025': Config(alpha=0.7, gamma=0.25, epochs=30),
    'exp3_a09_g01': Config(alpha=0.9, gamma=0.1, epochs=30),
}

# All experiments in order
ALL_EXPERIMENTS = {
    'baseline': BASELINE_CONFIG,
    **EXP1_CONFIGS,
    **EXP2_CONFIGS,
    **EXP3_CONFIGS,
}


# =============================================================================
# COLAB-SPECIFIC PATHS
# =============================================================================

def get_colab_paths():
    """Get paths for Google Colab environment."""
    return {
        'drive_root': '/content/drive/MyDrive/robust-ensemble-kd',
        'data_dir': '/content/drive/MyDrive/robust-ensemble-kd/data/waterbirds_v1.0',
        'checkpoint_dir': '/content/drive/MyDrive/robust-ensemble-kd/checkpoints',
        'log_dir': '/content/drive/MyDrive/robust-ensemble-kd/logs',
    }


def get_local_paths():
    """Get paths for local development."""
    return {
        'data_dir': './data/waterbirds_v1.0',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
    }
