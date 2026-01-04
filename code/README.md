# Robust Ensemble Knowledge Distillation

Extending AGRE-KD with class labels and feature distillation for improved group robustness.

## 🚀 Quick Start (Google Colab)

**Recommended**: Use the complete Colab notebook: `notebooks/colab_full_workflow.ipynb`

This notebook handles everything: setup, data download, teacher preparation, and experiments.

### Manual Setup

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git
cd robust-ensemble-kd/code
pip install torch torchvision wilds tqdm pandas numpy matplotlib scikit-learn

# 2. Download Waterbirds data
python -c "from data import download_waterbirds; download_waterbirds('./data')"

# 3. Download DFR checkpoints to teacher_checkpoints/ (see below)

# 4. Prepare debiased teachers (saves to same folder)
python prepare_teachers.py \
    --checkpoint_dir ./teacher_checkpoints \
    --data_dir ./data/waterbirds_v1.0

# 5. Run experiments
python train.py --mode student \
    --data_dir ./data/waterbirds_v1.0 \
    --teacher_paths ./teacher_checkpoints/teacher_*_debiased.pt \
    --alpha 0.7 --gamma 0.1 --exp_name exp3_combined
```

## 📁 Project Structure

```
code/
├── data.py              # Waterbirds dataloader with group labels
├── models.py            # ResNet with feature extraction hooks
├── losses.py            # KD, feature, AGRE-KD losses
├── eval.py              # WGA and per-group accuracy metrics
├── config.py            # All hyperparameters
├── train.py             # Teacher and student training loops
├── dfr.py               # Deep Feature Reweighting implementation
├── prepare_teachers.py  # Prepare debiased teachers from ERM checkpoints
├── requirements.txt     # Dependencies
└── notebooks/
    ├── colab_full_workflow.ipynb  # ⭐ Complete Colab workflow
    └── run_experiments.ipynb      # Alternative experiment runner
```

## 📋 Complete Colab Workflow

### Step 1: Setup (5 min)

```python
# Mount Drive & install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install -q wilds tqdm scikit-learn

# Clone repo
!git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git /content/repo
%cd /content/repo/code
```

### Step 2: Download DFR Checkpoints (~5 min)

The DFR authors provide pre-trained ERM checkpoints:

**Source**: https://drive.google.com/drive/folders/1OQ_oPPgxgK_7j_GCt71znyiRj6hqi_UW

1. Navigate to: `spurious_feature_learning/results/waterbirds_paper`
2. Download 3-5 checkpoint files (e.g., `erm_seed0.pt`, `erm_seed1.pt`, etc.)
3. Upload to your Google Drive: `MyDrive/robust-ensemble-kd/teacher_checkpoints/`

### Step 3: Prepare Teachers (~30 min)

```python
from prepare_teachers import colab_prepare_teachers

results = colab_prepare_teachers(
    checkpoint_dir='/content/drive/MyDrive/robust-ensemble-kd/teacher_checkpoints',
    data_dir='/content/drive/MyDrive/robust-ensemble-kd/data/waterbirds_v1.0',
    num_teachers=5
)
```

This adds debiased teachers to the same folder:
- `erm_seed0.pt` ... `erm_seed4.pt` → Biased (~70% WGA) - **you download these**
- `teacher_0_debiased.pt` ... `teacher_4_debiased.pt` → Debiased (~92% WGA) - **created by DFR**

### Step 4: Run Experiments (~2-3 hrs each)

```python
from config import Config
from train import train_student

# Load teachers
teachers = load_teachers(...)  # See notebook for details

# Baseline
config = Config(alpha=1.0, gamma=0.0, epochs=30)
train_student(config, teachers, exp_name='baseline')

# Experiment 1: Add class labels
config = Config(alpha=0.7, gamma=0.0, epochs=30)
train_student(config, teachers, exp_name='exp1_alpha07')

# Experiment 2: Feature distillation  
config = Config(alpha=1.0, gamma=0.1, epochs=30)
train_student(config, teachers, exp_name='exp2_gamma01')

# Experiment 3: Combined
config = Config(alpha=0.7, gamma=0.1, epochs=30)
train_student(config, teachers, exp_name='exp3_combined')
```

## 🔧 Usage Examples

### Load Data

```python
from data import get_waterbirds_loaders

loaders = get_waterbirds_loaders(
    root_dir='./data/waterbirds_v1.0',
    batch_size=32,
    augment=True
)
# loaders['train'], loaders['val'], loaders['test']
```

### Create Models

```python
from models import get_teacher_model, get_student_model, create_feature_adapter

# Teacher (ResNet-50)
teacher = get_teacher_model('resnet50', num_classes=2, pretrained=True)

# Student (ResNet-18)  
student = get_student_model('resnet18', num_classes=2, pretrained=True)

# Adapter for feature distillation (512 -> 2048)
adapter = create_feature_adapter('resnet18', 'resnet50', 'pooled')
```

### Extract Features

```python
# Forward pass with features
logits, features = teacher(images, return_features=True)
# features['pooled'] = penultimate features [B, 2048]
# features['layer4'] = spatial features [B, 2048, 7, 7]
```

### Apply DFR (Debiasing)

```python
from dfr import apply_dfr

# Transform biased model (~70% WGA) to debiased (~92% WGA)
apply_dfr(model, val_loader, device='cuda', method='sklearn', balance_type='group')
```

### Compute Losses

```python
from losses import AGREKDLoss

# AGRE-KD with class labels and features
loss_fn = AGREKDLoss(alpha=0.7, gamma=0.1, temperature=4.0,
                      student_dim=512, teacher_dim=2048)

# Compute teacher weights (gradient-based)
weights = loss_fn.compute_teacher_weights(student, teacher_logits_list, 
                                          biased_logits, student_logits)

# Compute loss
loss, loss_dict = loss_fn(student_logits, teacher_logits_list, labels,
                          student_features, teacher_features_list,
                          teacher_weights=weights)
```

### Evaluate

```python
from eval import compute_group_accuracies, print_results

results = compute_group_accuracies(model, loaders['test'])
print_results(results)
# Shows: per-group acc, WGA, average acc, accuracy gap
```

## 📊 Expected Results

| Method | Waterbirds WGA | Avg Acc |
|--------|----------------|---------|
| ERM (single model) | 68-72% | 97% |
| Deep Ensemble (3 models) | 75-80% | 96% |
| DFR | 91-93% | 94% |
| AGRE-KD (paper) | ~85-88% | 92% |
| **Your target** | ≥85% | ≥90% |

## 🔬 Your Three Experiments

### Experiment 1: Class Labels (α < 1, γ = 0)

**Hypothesis**: Adding ground-truth supervision alongside KD helps when teachers make mistakes.

```python
config = Config(alpha=0.7, gamma=0.0)  # Test α ∈ {0.5, 0.7, 0.9}
```

### Experiment 2: Feature Distillation (α = 1, γ > 0)

**Hypothesis**: Distilling penultimate features transfers more robust representations.

```python
config = Config(alpha=1.0, gamma=0.1)  # Test γ ∈ {0.1, 0.25}
```

### Experiment 3: Combined (α < 1, γ > 0)

**Hypothesis**: Both modifications together provide complementary benefits.

```python
config = Config(alpha=0.7, gamma=0.1)
```

## 💾 Google Drive Structure

```
MyDrive/robust-ensemble-kd/
├── data/
│   └── waterbirds_v1.0/           # Dataset (downloaded via WILDS)
├── teacher_checkpoints/            # All teachers in ONE folder
│   ├── erm_seed0.pt               # Biased (downloaded from DFR)
│   ├── erm_seed1.pt               # Biased (downloaded from DFR)
│   ├── erm_seed2.pt               # Biased (downloaded from DFR)
│   ├── teacher_0_debiased.pt      # Debiased (created by prepare_teachers.py)
│   ├── teacher_1_debiased.pt      # Debiased (created by prepare_teachers.py)
│   ├── teacher_2_debiased.pt      # Debiased (created by prepare_teachers.py)
│   └── ...
├── checkpoints/                    # Student training checkpoints
│   ├── student_baseline_best.pt
│   ├── student_exp1_alpha07_best.pt
│   └── ...
└── logs/
    └── experiment_results.json
```

## 🚨 Troubleshooting

**Colab disconnects**: Checkpoints saved every 5 epochs. Re-run cells 1-4, then resume.

**OOM errors**: Reduce `batch_size` to 16, use `torch.cuda.empty_cache()`

**Slow data loading**: Copy data to Colab local storage:
```bash
!cp -r /content/drive/MyDrive/.../data /content/data_local
```

**WILDS download fails**: Download manually from:
https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

**DFR checkpoints not found**: Download from:
https://drive.google.com/drive/folders/1OQ_oPPgxgK_7j_GCt71znyiRj6hqi_UW

## ⏱️ Time Estimates (T4 GPU)

| Task | Time |
|------|------|
| Setup & data download | 10 min |
| Prepare 5 teachers (DFR) | 30 min |
| Student training (30 epochs) | 2-3 hrs |
| Full experiment suite (6 configs) | ~15-20 hrs |

## 📚 References

- AGRE-KD: [arXiv:2411.14984](https://arxiv.org/abs/2411.14984)
- DFR: [arXiv:2204.02937](https://arxiv.org/abs/2204.02937) 
- DFR Checkpoints: [Google Drive](https://drive.google.com/drive/folders/1OQ_oPPgxgK_7j_GCt71znyiRj6hqi_UW)
- Group DRO: [github.com/kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)
- WILDS: [github.com/p-lambda/wilds](https://github.com/p-lambda/wilds)
