# Codebase Documentation: Robust Ensemble Knowledge Distillation

**Last Updated**: December 7, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Modules](#3-core-modules)
   - [config.py](#31-configpy---configuration-management)
   - [models.py](#32-modelspy---neural-network-architectures)
   - [losses.py](#33-lossespy---loss-functions)
   - [data.py](#34-datapy---data-loading)
   - [eval.py](#35-evalpy---evaluation-metrics)
   - [train.py](#36-trainpy---training-loops)
   - [dfr.py](#37-dfrpy---deep-feature-reweighting)
   - [prepare_teachers.py](#38-prepare_teacherspy---teacher-preparation)
4. [Notebooks](#4-notebooks)
5. [Architecture & Data Flow](#5-architecture--data-flow)
6. [Key Hyperparameters](#6-key-hyperparameters--experiments)
7. [Usage Examples](#7-usage-examples)

---

## 1. Project Overview

**Repository**: `robust-ensemble-kd`
**Purpose**: Extending AGRE-KD (Adaptive Group Robust Ensemble Knowledge Distillation) with class labels and feature distillation for improved group robustness on the Waterbirds dataset.

### Core Research Questions

1. **Experiment 1 (Class Labels)**: Does adding ground-truth supervision (α < 1) improve robustness?
2. **Experiment 2 (Feature Distillation)**: Does distilling intermediate features (γ > 0) help?
3. **Experiment 3 (Combined)**: Do these modifications provide complementary benefits?

### Training Pipeline Summary

```
ERM Teachers (biased, ~70% WGA)
        ↓ DFR debiasing
Debiased Teachers (~92% WGA)
        ↓ AGRE-KD distillation
Student Model (target: 85%+ WGA)
```

---

## 2. Directory Structure

```
light-code/
├── config.py                    # Hyperparameter configuration
├── data.py                      # Waterbirds dataset loader
├── models.py                    # ResNet architectures with feature extraction
├── losses.py                    # Loss functions (KD, feature, AGRE-KD)
├── eval.py                      # Evaluation metrics and logging
├── train.py                     # Training loops for teachers and students
├── dfr.py                       # Deep Feature Reweighting implementation
├── prepare_teachers.py          # Teacher preparation script
├── requirements.txt             # Package dependencies
├── README.md                    # Quick start guide
└── notebooks/
    ├── 01_setup.ipynb           # Initial setup and exploration
    ├── 02_experiments.ipynb     # Main experiment runner
    ├── 03_seed_experiments.ipynb # Multi-seed experiments + ablations
    └── run_experiments.ipynb    # Additional experiment runs
```

---

## 3. Core Modules

### 3.1 `config.py` - Configuration Management

**Purpose**: Centralized hyperparameter management via dataclass.

#### Key Class: `Config`

```python
@dataclass
class Config:
    # Data
    data_dir: str
    batch_size: int = 128
    num_workers: int = 2
    augment: bool = True

    # Model
    teacher_arch: str = 'resnet50'
    student_arch: str = 'resnet18'
    num_classes: int = 2
    pretrained: bool = True

    # Training
    epochs: int = 50
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = 'cosine'

    # Distillation
    temperature: float = 4.0
    alpha: float = 1.0      # KD weight (1.0 = pure KD)
    gamma: float = 0.0      # Feature distillation weight

    # Checkpointing
    checkpoint_dir: str
    save_freq: int = 10

    # Misc
    seed: int = 42
    device: str = 'cuda'
```

#### Pre-configured Experiments

| Config | α | γ | Description |
|--------|---|---|-------------|
| `BASELINE_CONFIG` | 1.0 | 0.0 | Pure KD (no class labels) |
| `EXP1_CONFIGS` | 0.5-0.9 | 0.0 | Class labels experiments |
| `EXP2_CONFIGS` | 1.0 | 0.1-0.25 | Feature distillation |
| `EXP3_CONFIGS` | <1.0 | >0.0 | Combined approach |

#### Path Helpers

- `get_colab_paths()`: Returns paths for Google Colab environment
- `get_local_paths()`: Returns paths for local development

---

### 3.2 `models.py` - Neural Network Architectures

**Purpose**: Define ResNet models with intermediate feature extraction.

#### Key Class: `ResNetWithFeatures`

```python
class ResNetWithFeatures(nn.Module):
    """ResNet wrapper that captures intermediate features."""

    def __init__(self, arch='resnet50', num_classes=2, pretrained=True):
        # Supports: 'resnet18' (512-d), 'resnet50' (2048-d)

    def forward(self, x, return_features=False):
        """
        Returns:
            logits: [B, num_classes]
            features (if return_features=True): Dict with:
                - 'layer1': [B, C1, H1, W1]
                - 'layer2': [B, C2, H2, W2]
                - 'layer3': [B, C3, H3, W3]
                - 'layer4': [B, C4, 7, 7]  # Penultimate spatial
                - 'pooled': [B, D]          # After global avg pool
        """
```

#### Feature Adapter

```python
class FeatureAdapter(nn.Module):
    """Maps student features to teacher dimensions."""

    def __init__(self, student_dim, teacher_dim, spatial=False):
        # Uses Conv2d if spatial=True, else Linear
        # Identity if dimensions match
```

#### Factory Functions

| Function | Purpose |
|----------|---------|
| `get_teacher_model(arch, num_classes, pretrained)` | Create teacher (ResNet-50) |
| `get_student_model(arch, num_classes, pretrained)` | Create student (ResNet-18) |
| `create_feature_adapter(student_arch, teacher_arch, layer)` | Create dimension adapter |

#### Checkpoint Loading

```python
def load_teacher_checkpoint(model, path, strict=False):
    """
    Robust checkpoint loading that handles:
    - Multiple key formats: 'model', 'state_dict', 'model_state_dict'
    - DataParallel 'module.' prefix
    - Various backbone prefixes
    """
```

---

### 3.3 `losses.py` - Loss Functions

**Purpose**: Implement knowledge distillation and ensemble loss functions.

#### Loss Function Hierarchy

```
               ┌─────────────────┐
               │    KDLoss       │  Standard KL divergence
               └────────┬────────┘
                        │
               ┌────────┴────────┐
               │FeatureDistill   │  MSE on intermediate features
               │     Loss        │
               └────────┬────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
│ CombinedDistill│ │EnsembleKD │ │   AGREKDLoss  │
│     Loss      │ │   Loss    │ │               │
└───────────────┘ └───────────┘ └───────────────┘
     Simple          AVER         Gradient-weighted
```

#### `KDLoss`

```python
class KDLoss(nn.Module):
    """Standard Knowledge Distillation loss."""

    def __init__(self, temperature=4.0):
        # L_KD = KL(softmax(z_s/T) || softmax(z_t/T)) × T²
```

#### `FeatureDistillationLoss`

```python
class FeatureDistillationLoss(nn.Module):
    """Feature-level distillation with dimension adaptation."""

    def __init__(self, student_dim, teacher_dim, spatial=False):
        # L_feat = MSE(adapter(f_student), f_teacher)
        # Handles dimension mismatch via adapter
        # Handles spatial size mismatch via adaptive pooling
```

#### `AGREKDLoss` (Main Loss Function)

```python
class AGREKDLoss(nn.Module):
    """Adaptive Group Robust Ensemble Knowledge Distillation."""

    def __init__(self, temperature=4.0, alpha=1.0, gamma=0.0):
        # alpha: Weight for KD vs CE (1.0 = pure KD)
        # gamma: Weight for feature distillation

    def compute_teacher_weights(self, student, teachers, biased_model, images):
        """
        Gradient-based per-teacher weighting.

        Formula: W_t(x) = 1 - cos_sim(grad_teacher_t, grad_biased)

        Teachers agreeing with biased model are downweighted.
        """

    def forward(self, student_logits, teacher_logits_list, labels,
                student_features=None, teacher_features=None, weights=None):
        """
        Compute combined loss.

        L_total = (1-α)×L_CE + α×L_KD + γ×L_feat

        Returns: (loss, loss_dict)
        """
```

#### Loss Dict Contents

```python
{
    'ce': float,      # Cross-entropy (if alpha < 1)
    'kd': float,      # Knowledge distillation
    'feat': float,    # Feature distillation (if gamma > 0)
    'total': float,   # Sum of weighted components
    'weights': list   # Per-teacher weights (AGRE-KD only)
}
```

---

### 3.4 `data.py` - Data Loading

**Purpose**: Waterbirds dataset with group labels for spurious correlation research.

#### Waterbirds Groups

| Group | Label | Background | Type | Size |
|-------|-------|------------|------|------|
| 0 | Landbird | Land | Majority | Large |
| 1 | Landbird | Water | Minority | Small |
| 2 | Waterbird | Land | Minority | **Smallest (hardest!)** |
| 3 | Waterbird | Water | Majority | Large |

#### `WaterbirdsDataset`

```python
class WaterbirdsDataset(Dataset):
    """PyTorch dataset for Waterbirds with group labels."""

    def __init__(self, root_dir, split='train', transform=None):
        # Loads from: waterbird_complete95_forest2water2/
        # Metadata: img_filename, y (class), place (location), split

    def __getitem__(self, idx):
        # Returns dict: {'image', 'label', 'group', 'index'}
```

#### Data Transforms

```python
def get_transforms(train=True, augment=True):
    """
    Train with augmentation:
        - RandomResizedCrop(224)
        - RandomHorizontalFlip

    Eval:
        - Resize(256) → CenterCrop(224)

    Both:
        - Normalize with ImageNet mean/std
    """
```

#### Factory Function

```python
def get_waterbirds_loaders(root_dir, batch_size=32, num_workers=4, augment=True):
    """Returns dict: {'train': loader, 'val': loader, 'test': loader}"""
```

---

### 3.5 `eval.py` - Evaluation Metrics

**Purpose**: Compute group robustness metrics.

#### Main Evaluation Function

```python
def compute_group_accuracies(model, dataloader, device='cuda', verbose=True):
    """
    Compute per-group and aggregate accuracies.

    Returns dict:
        'group_accs': {0: acc, 1: acc, 2: acc, 3: acc}
        'wga': float          # Worst-group accuracy (PRIMARY METRIC)
        'worst_group': int    # Index of worst group
        'avg_acc': float      # Overall accuracy
        'acc_gap': float      # Max - Min accuracy
        'group_counts': dict  # Samples per group
    """
```

#### Utility Functions

| Function | Purpose |
|----------|---------|
| `print_results(results, title)` | Pretty-print evaluation results |
| `evaluate_ensemble(models, dataloader, device)` | Evaluate averaged ensemble |

#### `MetricLogger`

```python
class MetricLogger:
    """Simple history tracking for training."""

    def log(self, metrics: dict)      # Append to history
    def get_last(self, key: str)      # Most recent value
    def get_best(self, key: str, mode='max')  # Best value
    def to_dict()                      # Export as dictionary
```

---

### 3.6 `train.py` - Training Loops

**Purpose**: Train teachers (ERM) and students (distillation).

#### Teacher Training

```python
def train_teacher(config, checkpoint_path=None):
    """
    Train teacher with ERM (cross-entropy loss).

    Args:
        config: Config object with hyperparameters
        checkpoint_path: Optional path to resume from

    Returns:
        (model, history_dict)

    Saves:
        - Best checkpoint (by val WGA)
        - Periodic checkpoints every N epochs
    """
```

#### Student Training

```python
def train_student(config, teachers, biased_model=None, exp_name='student',
                  use_agre=True, checkpoint_path=None):
    """
    Train student via knowledge distillation.

    Args:
        config: With alpha, gamma, temperature
        teachers: List of teacher models (eval mode)
        biased_model: Reference for AGRE-KD (defaults to teachers[0])
        exp_name: For checkpoint naming
        use_agre: True=gradient weights, False=uniform (AVER)
        checkpoint_path: Optional path to resume from

    Returns:
        (student_model, history_dict, test_results)

    Saves:
        - student_{exp_name}_best.pt
        - student_{exp_name}_latest.pt
        - student_{exp_name}_log.json
    """
```

#### Training Process (Student)

1. Forward student with gradients
2. Forward teachers without gradients
3. If `use_agre=True`: Compute gradient-based teacher weights
4. Compute combined loss: `(1-α)×CE + α×KD + γ×Feat`
5. Backprop and update student + adapter (if used)

#### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'adapter_state_dict': dict or None,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_wga': float,
    'config': dict,
    'group_accs': dict
}
```

---

### 3.7 `dfr.py` - Deep Feature Reweighting

**Purpose**: Transform biased ERM models (~70% WGA) into debiased teachers (~92% WGA).

#### DFR Concept

1. Train model with ERM → learns biased features
2. Freeze all layers except classifier
3. Retrain classifier on **balanced** validation subset
4. Result: Debiased predictions using same features

#### Key Functions

```python
def extract_features(model, dataloader, device, verbose=True):
    """Extract penultimate pooled features from model."""
    # Returns: (features [N, D], labels [N], groups [N])

def create_balanced_subset(features, labels, groups, balance_type='group'):
    """
    Create balanced training data.

    balance_type:
        'group': Balance all 4 groups (aggressive)
        'class': Balance 2 classes only
    """

def train_dfr_sklearn(features, labels, C=1.0, class_weight='balanced'):
    """Fast logistic regression (~10 seconds)."""

def apply_dfr(model, val_loader, device='cuda', method='sklearn',
              balance_type='group', C=1.0, verbose=True):
    """
    Main function: Transform biased → debiased.

    Steps:
        1. Extract features from val_loader
        2. Create balanced subset
        3. Train new classifier
        4. Replace model's FC layer in-place

    Returns: Modified model
    """
```

#### Batch Processing

```python
def apply_dfr_to_teachers(teachers, val_loader, device, method='sklearn'):
    """Apply DFR to multiple teachers."""
```

---

### 3.8 `prepare_teachers.py` - Teacher Preparation

**Purpose**: Batch process ERM checkpoints into debiased teachers.

#### Workflow

```
erm_seed42.pt → [Load] → [Evaluate biased] → [Apply DFR] → [Evaluate debiased] → teacher_0_debiased.pt
erm_seed43.pt → [Load] → [Evaluate biased] → [Apply DFR] → [Evaluate debiased] → teacher_1_debiased.pt
...
```

#### Key Functions

```python
def find_erm_checkpoints(checkpoint_dir):
    """Find all erm_seed*.pt files."""
    # Returns: List of dicts with path, seed, filename

def prepare_single_teacher(erm_info, loaders, device):
    """
    5-step process:
        1. Load model
        2. Evaluate biased version
        3. Apply DFR
        4. Evaluate debiased version
        5. Save checkpoint with metadata
    """

def prepare_all_teachers(checkpoint_dir, data_dir, num_teachers, device='cuda'):
    """Batch process and print summary."""

def colab_prepare_teachers(checkpoint_dir, data_dir, num_teachers=5):
    """Convenience wrapper for Google Colab."""
```

#### Expected Results

| Model | Biased WGA | Debiased WGA | Improvement |
|-------|------------|--------------|-------------|
| Teacher 1 | ~70% | ~92% | +22% |
| Teacher 2 | ~70% | ~92% | +22% |
| ... | ... | ... | ... |

---

## 4. Notebooks

### `01_setup.ipynb`
- Environment setup and dependency installation
- Data download (WILDS Waterbirds)
- Basic exploration of data structure
- Verification of model loading

### `02_experiments.ipynb`
- Main experiment runner (236 KB)
- Loads teachers and trains students
- Logs results for each configuration
- Contains visualization and analysis

### `03_seed_experiments.ipynb`
- Multi-seed experiments for statistical significance
- Gamma ablation: γ ∈ {0.05, 0.25, 0.50, 0.75, 1.00}
- AVER vs AGRE-KD comparison
- Computer-based experiment splitting

### `run_experiments.ipynb`
- Additional experiment runs
- Quick iteration on configurations

---

## 5. Architecture & Data Flow

### Training Pipeline

```
                    ┌──────────────────┐
                    │   Waterbirds     │
                    │    Dataset       │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │   data.py        │
                    │   - Train/val/test splits
                    │   - Group labels (0-3)
                    │   - Augmentation
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────┴─────────┐         ┌─────────┴─────────┐
    │  Teacher Training │         │  Student Training │
    │  (train_teacher)  │         │  (train_student)  │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              │ ERM (CE loss)               │ AGRE-KD loss
              │                             │
    ┌─────────┴─────────┐                   │
    │  Biased Teachers  │                   │
    │  (WGA ~70%)       │                   │
    └─────────┬─────────┘                   │
              │                             │
              │ DFR (dfr.py)                │
              │                             │
    ┌─────────┴─────────┐                   │
    │ Debiased Teachers │───────────────────┤
    │  (WGA ~92%)       │                   │
    └───────────────────┘                   │
                                            │
                              ┌─────────────┴─────────────┐
                              │      Student Model        │
                              │      (ResNet-18)          │
                              └─────────────┬─────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              │    Evaluation (eval.py)   │
                              │    - Per-group accuracy   │
                              │    - WGA (primary metric) │
                              └───────────────────────────┘
```

### Model Architecture

```
Input Image [B, 3, 224, 224]
              │
    ┌─────────┴─────────┐
    │       Stem        │
    │  Conv→BN→ReLU→Pool│
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │      Layer1       │ ──→ features['layer1']
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │      Layer2       │ ──→ features['layer2']
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │      Layer3       │ ──→ features['layer3']
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │      Layer4       │ ──→ features['layer4'] [B, C, 7, 7]
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │   Global AvgPool  │ ──→ features['pooled'] [B, D]
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │    FC Classifier  │
    └─────────┬─────────┘
              │
        Logits [B, 2]
```

### Loss Computation (AGRE-KD)

```
Student Forward
      │
      ├──→ student_logits
      │
      └──→ student_features (if γ > 0)
              │
              │         ┌─────────────────┐
              │         │ Teacher 1       │──→ logits_1, features_1
              │         ├─────────────────┤
              ├────────→│ Teacher 2       │──→ logits_2, features_2
              │         ├─────────────────┤
              │         │ Teacher N       │──→ logits_N, features_N
              │         └─────────────────┘
              │                   │
              │    ┌──────────────┴──────────────┐
              │    │  Compute Teacher Weights     │
              │    │  (if use_agre=True)          │
              │    │                              │
              │    │  w_t = 1 - cos_sim(          │
              │    │    grad_teacher_t,           │
              │    │    grad_biased               │
              │    │  )                           │
              │    └──────────────┬──────────────┘
              │                   │
              │                   ▼
              │         weights = softmax(w)
              │                   │
              ▼                   ▼
    ┌─────────────────────────────────────────┐
    │           Combined Loss                  │
    │                                          │
    │  L = (1-α)×CE + α×Σ(w_t×KD_t) + γ×Feat  │
    │                                          │
    └─────────────────────────────────────────┘
```

---

## 6. Key Hyperparameters & Experiments

### Configuration Space

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `alpha` | 1.0 | [0.0, 1.0] | KD vs CE weight (1.0 = pure KD) |
| `gamma` | 0.0 | [0.0, 1.0] | Feature distillation weight |
| `temperature` | 4.0 | [2, 10] | KD softness (higher = softer) |
| `lr` | 0.001 | [1e-4, 1e-2] | Learning rate |
| `epochs` | 30 | [20, 100] | Training epochs |
| `batch_size` | 128 | [32, 128] | Batch size |

### Experiment Matrix

| Experiment | α | γ | Hypothesis |
|------------|---|---|------------|
| **Baseline** | 1.0 | 0.0 | Pure KD reference |
| **Exp1** | 0.7-0.9 | 0.0 | Ground truth helps when teachers disagree |
| **Exp2** | 1.0 | 0.05-1.0 | Intermediate features are more robust |
| **Exp3** | 0.9 | 0.25 | Complementary benefits |

### Gamma Ablation Points

| γ | Purpose |
|---|---------|
| 0.00 | Baseline (no feature distillation) |
| 0.05 | Very weak feature signal |
| 0.25 | Moderate feature weight |
| 0.50 | Equal to base KD |
| 0.75 | Strong feature weight |
| 1.00 | Equal KD and feature weights |

---

## 7. Usage Examples

### Load Data

```python
from data import get_waterbirds_loaders

loaders = get_waterbirds_loaders(
    root_dir='./data/waterbirds_v1.0',
    batch_size=32,
    num_workers=4,
    augment=True
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### Create Models

```python
from models import get_teacher_model, get_student_model, create_feature_adapter

# Teacher (ResNet-50, 2048-d features)
teacher = get_teacher_model('resnet50', num_classes=2, pretrained=True)

# Student (ResNet-18, 512-d features)
student = get_student_model('resnet18', num_classes=2, pretrained=True)

# Adapter for feature distillation (512 → 2048)
adapter = create_feature_adapter('resnet18', 'resnet50', layer='layer4')
```

### Load Checkpoint

```python
from models import load_teacher_checkpoint

model = get_teacher_model('resnet50', num_classes=2, pretrained=False)
load_teacher_checkpoint(model, 'checkpoints/teacher_0_debiased.pt')
model.eval()
```

### Apply DFR

```python
from dfr import apply_dfr

# Transform biased → debiased (in-place)
apply_dfr(
    model=teacher,
    val_loader=loaders['val'],
    device='cuda',
    method='sklearn',
    balance_type='group',
    verbose=True
)
```

### Train Student

```python
from config import Config
from train import train_student

config = Config(
    data_dir='./data/waterbirds_v1.0',
    checkpoint_dir='./checkpoints',
    alpha=0.9,
    gamma=0.25,
    epochs=30
)

student, history, results = train_student(
    config=config,
    teachers=teachers,
    biased_model=biased_model,
    exp_name='exp3_combined',
    use_agre=True
)

print(f"Test WGA: {results['wga']*100:.2f}%")
```

### Evaluate Model

```python
from eval import compute_group_accuracies, print_results

results = compute_group_accuracies(
    model=student,
    dataloader=loaders['test'],
    device='cuda',
    verbose=True
)

print_results(results, title="Student Evaluation")
# Output:
# Group 0 (Landbird+Land):  96.2%
# Group 1 (Landbird+Water): 88.5%
# Group 2 (Waterbird+Land): 85.1%  ← Worst group
# Group 3 (Waterbird+Water): 94.8%
# ---
# Worst-Group Accuracy: 85.1%
# Average Accuracy: 93.4%
```

### Create Custom Loss

```python
from losses import AGREKDLoss

criterion = AGREKDLoss(
    temperature=4.0,
    alpha=0.9,      # 10% CE, 90% KD
    gamma=0.25      # 25% feature distillation
).to(device)

loss, loss_dict = criterion(
    student_logits=s_logits,
    teacher_logits_list=[t1_logits, t2_logits, t3_logits],
    labels=labels,
    student_features=s_feat,
    teacher_features=avg_t_feat,
    weights=teacher_weights  # From compute_teacher_weights()
)
```

---

## Appendix: File Summary

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `config.py` | ~117 | 3.1 KB | Configuration management |
| `models.py` | ~335 | 11.6 KB | Model architectures + loading |
| `losses.py` | ~493 | 18.4 KB | Loss functions |
| `data.py` | ~203 | 7.0 KB | Data loading |
| `eval.py` | ~240 | 7.5 KB | Evaluation metrics |
| `train.py` | ~483 | 18.6 KB | Training loops |
| `dfr.py` | ~413 | 14.1 KB | DFR debiasing |
| `prepare_teachers.py` | ~328 | 12.7 KB | Teacher preparation |

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
wilds>=2.0.0
scikit-learn>=1.3.0
```
