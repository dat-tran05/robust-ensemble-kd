# Robust Ensemble Knowledge Distillation

Extending AGRE-KD with class labels and feature distillation for improved group robustness.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git
cd robust-ensemble-kd
pip install torch torchvision wilds tqdm pandas numpy matplotlib

# 2. Download Waterbirds data
python -c "from data import download_waterbirds; download_waterbirds('./data')"

# 3. Test data loading
python data.py --data_dir ./data/waterbirds_v1.0
```

## 📁 Project Structure

```
robust-ensemble-kd/
├── data.py      # Waterbirds dataloader with group labels
├── models.py    # ResNet with feature extraction hooks
├── losses.py    # KD, feature, and combined losses
├── eval.py      # WGA and per-group accuracy metrics
├── config.py    # All hyperparameters
├── train.py     # Training loops (create during Day 1-2)
└── notebooks/
    ├── 01_train_teachers.ipynb
    ├── 02_train_student.ipynb
    └── 03_analysis.ipynb
```

## 📋 Implementation Plan

### Day 1: Setup + Teachers

| Task | Person | Time |
|------|--------|------|
| Clone repo, download data | Both | 30 min |
| Test data.py, models.py | A | 1 hr |
| Train Teacher 1 (seed=42) | A | ~2 hrs |
| Train Teacher 2 (seed=43) | B | ~2 hrs |
| Train Teacher 3 (seed=44) | B | ~2 hrs |

**Checkpoint**: 3 trained teachers saved to Google Drive

### Day 2: Distillation Setup + Baseline

| Task | Person | Time |
|------|--------|------|
| Implement train.py student loop | A | 2 hrs |
| Verify loss functions work | A | 1 hr |
| Run baseline (α=1, γ=0) | B | 2 hrs |
| Start Exp 1: α=0.7 | B | 2 hrs |

**Checkpoint**: Baseline WGA established

### Day 3: Core Experiments

| Task | Person | Time |
|------|--------|------|
| Exp 1 ablations: α ∈ {0.5, 0.9} | A | 3 hrs |
| Exp 2: γ=0.1, γ=0.25 | B | 3 hrs |
| Exp 3: α=0.7, γ=0.1 | A | 2 hrs |

**Checkpoint**: All 3 experiments on Waterbirds complete

### Day 4: Analysis + Buffer

| Task | Person | Time |
|------|--------|------|
| Compile results | Both | 2 hrs |
| Create visualizations | A | 2 hrs |
| Run additional ablations if needed | B | 4 hrs |

### Day 5: Write-up

| Task | Person | Time |
|------|--------|------|
| Draft blog post | A | 4 hrs |
| Create figures/tables | B | 2 hrs |
| Review and finalize | Both | 2 hrs |

---

## 🔧 Usage Examples

### 1. Load Data

```python
from data import get_waterbirds_loaders

loaders = get_waterbirds_loaders(
    root_dir='./data/waterbirds_v1.0',
    batch_size=32,
    augment=True
)

# loaders['train'], loaders['val'], loaders['test']
```

### 2. Create Models

```python
from models import get_teacher_model, get_student_model, create_feature_adapter

# Teacher (ResNet-50)
teacher = get_teacher_model('resnet50', num_classes=2, pretrained=True)

# Student (ResNet-18)  
student = get_student_model('resnet18', num_classes=2, pretrained=True)

# Adapter for feature distillation (512 -> 2048)
adapter = create_feature_adapter('resnet18', 'resnet50', 'pooled')
```

### 3. Extract Features

```python
# Forward pass with features
logits, features = teacher(images, return_features=True)
# features['pooled'] = penultimate features [B, 2048]
# features['layer4'] = spatial features [B, 2048, 7, 7]
```

### 4. Compute Losses

```python
from losses import CombinedDistillationLoss

# Experiment 1: Add class labels (α < 1)
loss_fn = CombinedDistillationLoss(alpha=0.7, gamma=0.0)

# Experiment 2: Feature distillation (γ > 0)
loss_fn = CombinedDistillationLoss(alpha=1.0, gamma=0.1, 
                                    student_dim=512, teacher_dim=2048)

# Experiment 3: Combined
loss_fn = CombinedDistillationLoss(alpha=0.7, gamma=0.1,
                                    student_dim=512, teacher_dim=2048)

# Usage
loss, loss_dict = loss_fn(
    student_logits, teacher_logits, labels,
    student_features, teacher_features  # For γ > 0
)
```

### 5. Evaluate

```python
from eval import compute_group_accuracies, print_results

results = compute_group_accuracies(model, loaders['test'])
print_results(results)
# Shows: per-group acc, WGA, average acc, accuracy gap
```

---

## 📊 Expected Results (Baselines to Beat)

| Method | Waterbirds WGA | Avg Acc |
|--------|----------------|---------|
| ERM (single model) | 68-72% | 97% |
| Deep Ensemble (3 models) | 75-80% | 96% |
| JTT | 86-87% | 93% |
| DFR | 91-93% | 94% |
| AGRE-KD (paper) | ~85-88% | 92% |

**Your target**: Match or exceed AGRE-KD baseline (~85% WGA)

---

## 🔬 Your Three Experiments

### Experiment 1: Class Labels (α < 1, γ = 0)

**Hypothesis**: Adding ground-truth supervision alongside KD helps when teachers are biased.

```python
# Test α ∈ {0.5, 0.7, 0.9}
config = Config(alpha=0.7, gamma=0.0)
```

### Experiment 2: Feature Distillation (α = 1, γ > 0)

**Hypothesis**: Distilling penultimate features transfers more robust representations.

```python
# Test γ ∈ {0.1, 0.25}
config = Config(alpha=1.0, gamma=0.1)
```

### Experiment 3: Combined (α < 1, γ > 0)

**Hypothesis**: Both modifications together provide complementary benefits.

```python
config = Config(alpha=0.7, gamma=0.1)
```

---

## 💾 Google Colab Setup

Paste this in the first cell of every notebook:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
DRIVE_PATH = '/content/drive/MyDrive/robust-ensemble-kd'
import os
os.makedirs(f'{DRIVE_PATH}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_PATH}/data', exist_ok=True)

# Clone repo
!git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git /content/repo
%cd /content/repo

# Install deps
!pip install -q wilds tqdm

# Add to path
import sys
sys.path.insert(0, '/content/repo')

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🚨 Troubleshooting

**Colab disconnects**: Checkpoint every 5-10 epochs to Google Drive

**OOM errors**: Reduce batch_size to 16, use `torch.cuda.empty_cache()`

**Slow data loading**: Copy data to Colab local storage:
```bash
!cp -r /content/drive/MyDrive/robust-ensemble-kd/data /content/data_local
```

**WILDS download fails**: Download manually from:
https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

---

## 📚 References

- AGRE-KD: [arXiv:2411.14984](https://arxiv.org/abs/2411.14984)
- DFR: [arXiv:2204.02937](https://arxiv.org/abs/2204.02937) 
- Group DRO: [github.com/kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)
- WILDS: [github.com/p-lambda/wilds](https://github.com/p-lambda/wilds)
