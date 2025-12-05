# Implementation Guide: Robust Ensemble Knowledge Distillation

This guide covers how to set up your GitHub repository, organize your codebase, and run experiments on Google Colab efficiently with a 2-person team over 5 days.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [GitHub Repository Structure](#2-github-repository-structure)
3. [Google Colab Workflow](#3-google-colab-workflow)
4. [Step-by-Step Implementation Plan](#4-step-by-step-implementation-plan)
5. [Code Modules to Implement](#5-code-modules-to-implement)
6. [Training Pipeline](#6-training-pipeline)
7. [Checkpointing Strategy](#7-checkpointing-strategy)
8. [Team Coordination](#8-team-coordination)
9. [Debugging and Common Issues](#9-debugging-and-common-issues)

---

## 1. Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   GitHub Repository              Google Drive              Colab     │
│   ┌──────────────┐              ┌──────────────┐      ┌──────────┐  │
│   │ • Code       │   sync       │ • Checkpoints│      │ • GPU    │  │
│   │ • Configs    │ ◄──────────► │ • Datasets   │ ◄──► │ • Runtime│  │
│   │ • Notebooks  │              │ • Logs       │      │ • Train  │  │
│   └──────────────┘              └──────────────┘      └──────────┘  │
│         │                              │                     │       │
│         └──────────────────────────────┴─────────────────────┘       │
│                    Both team members access all three                │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principle**: Code lives in GitHub (version controlled), data/checkpoints live in Google Drive (persistent storage), execution happens in Colab (GPU access).

---

## 2. GitHub Repository Structure

Create a new repository with this structure:

```
robust-ensemble-kd/
├── README.md                    # Project overview, setup instructions
├── requirements.txt             # Dependencies
├── configs/
│   ├── waterbirds.yaml          # Dataset-specific configs
│   └── celeba.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── waterbirds.py        # Waterbirds dataloader with group labels
│   │   └── celeba.py            # CelebA dataloader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py            # ResNet with feature extraction hooks
│   │   └── adapters.py          # Feature dimension adapters
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── kd_loss.py           # Standard KD loss
│   │   ├── feature_loss.py      # Feature distillation loss
│   │   └── agre_kd.py           # AGRE-KD weighting scheme
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_teacher.py     # Teacher training script
│   │   ├── train_student.py     # Student distillation script
│   │   └── dfr.py               # DFR last-layer retraining
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # WGA, per-group accuracy
│       ├── checkpointing.py     # Save/load utilities
│       └── logging.py           # Experiment logging
├── notebooks/
│   ├── 01_train_teachers.ipynb  # Colab notebook for teacher training
│   ├── 02_dfr_retraining.ipynb  # DFR debiasing
│   ├── 03_train_student.ipynb   # Student distillation experiments
│   └── 04_analysis.ipynb        # Results analysis and visualization
├── scripts/
│   ├── setup_colab.py           # Colab environment setup
│   └── download_data.py         # Dataset download script
└── results/
    └── .gitkeep                 # Placeholder for results (stored in Drive)
```

### Setting Up the Repository

```bash
# On your local machine or in Colab
git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git
cd robust-ensemble-kd

# Create the directory structure
mkdir -p src/{data,models,losses,training,utils} notebooks scripts configs results
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch src/losses/__init__.py src/training/__init__.py src/utils/__init__.py
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
PyYAML>=6.0
wilds>=2.0.0
```

---

## 3. Google Colab Workflow

### 3.1 Initial Setup Cell (Run First in Every Notebook)

Create this as the first cell in every Colab notebook:

```python
# ============================================
# SETUP CELL - Run this first!
# ============================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
DRIVE_PATH = '/content/drive/MyDrive/robust-ensemble-kd'
REPO_PATH = '/content/robust-ensemble-kd'
CHECKPOINT_PATH = f'{DRIVE_PATH}/checkpoints'
DATA_PATH = f'{DRIVE_PATH}/data'
LOG_PATH = f'{DRIVE_PATH}/logs'

# Create directories in Drive (persistent storage)
import os
for path in [CHECKPOINT_PATH, DATA_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

# Clone/update repository
if os.path.exists(REPO_PATH):
    %cd {REPO_PATH}
    !git pull
else:
    !git clone https://github.com/YOUR_USERNAME/robust-ensemble-kd.git {REPO_PATH}
    %cd {REPO_PATH}

# Install dependencies
!pip install -q wilds tqdm pyyaml

# Add repo to Python path
import sys
sys.path.insert(0, REPO_PATH)

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

print("\n✅ Setup complete!")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Data: {DATA_PATH}")
print(f"📁 Logs: {LOG_PATH}")
```

### 3.2 Recommended Notebook Organization

| Notebook | Purpose | Who Runs It |
|----------|---------|-------------|
| `01_train_teachers.ipynb` | Train 3 ERM teachers | Person A (Day 1-2) |
| `02_dfr_retraining.ipynb` | Apply DFR to create debiased teachers | Person B (Day 2) |
| `03_train_student.ipynb` | Run all distillation experiments | Both (Day 2-4) |
| `04_analysis.ipynb` | Generate figures and tables | Both (Day 5) |

### 3.3 Pushing Changes from Colab to GitHub

```python
# Configure git (run once per session)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Stage, commit, and push changes
!git add .
!git commit -m "Add experiment results from Colab"
!git push
```

**Note**: For pushing to private repos, you'll need a Personal Access Token:
```python
# Use token instead of password
!git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/robust-ensemble-kd.git
```

---

## 4. Step-by-Step Implementation Plan

### Day 1: Setup + Teacher Training

**Person A (Setup Lead)**:
1. Create GitHub repository with structure above
2. Implement `src/data/waterbirds.py` dataloader
3. Implement `src/models/resnet.py` with feature hooks
4. Start training Teacher 1

**Person B (Training Lead)**:
1. Set up Google Drive folder structure
2. Download Waterbirds dataset to Drive
3. Start training Teacher 2 and Teacher 3 (parallel on different account)

**End of Day 1 Checkpoint**:
- [ ] Repository created and shared
- [ ] Waterbirds data downloaded
- [ ] At least 2 teachers training or completed

### Day 2: Complete Teachers + Start Experiments

**Person A**:
1. Implement `src/losses/agre_kd.py` (gradient-based weighting)
2. Implement `src/losses/feature_loss.py` (penultimate layer MSE)
3. Apply DFR to trained teachers

**Person B**:
1. Implement `src/training/train_student.py`
2. Run baseline: AGRE-KD (α=1, γ=0)
3. Start Experiment 1: Class labels (α=0.7, γ=0)

**End of Day 2 Checkpoint**:
- [ ] All 3 teachers trained and saved
- [ ] DFR applied to create debiased teachers
- [ ] AGRE-KD baseline running

### Day 3: Core Experiments

**Person A**:
1. Run Experiment 2: Feature distillation (α=1, γ=0.1)
2. Run Experiment 2 variant: (α=1, γ=0.25)

**Person B**:
1. Complete Experiment 1 ablations: α ∈ {0.5, 0.7, 0.9}
2. Run Experiment 3: Combined (α=0.7, γ=0.1)

**End of Day 3 Checkpoint**:
- [ ] Experiments 1-3 completed on Waterbirds
- [ ] Results logged and saved

### Day 4: Analysis + CelebA (if time)

**Person A**:
1. Run additional ablations if needed
2. Start CelebA teacher training (or subsample)

**Person B**:
1. Compile results into analysis notebook
2. Create per-group accuracy visualizations

**End of Day 4 Checkpoint**:
- [ ] All Waterbirds results finalized
- [ ] Figures drafted
- [ ] CelebA started (optional)

### Day 5: Write-up

**Both**:
1. Write blog post sections
2. Create final figures and tables
3. Review and polish
4. Submit!

---

## 5. Code Modules to Implement

### 5.1 Waterbirds Dataloader (`src/data/waterbirds.py`)

```python
"""Waterbirds dataset with group labels."""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset with group labels for spurious correlation research.
    
    Groups:
        0: landbird on land (majority)
        1: landbird on water (minority)
        2: waterbird on land (minority - hardest)
        3: waterbird on water (majority)
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._default_transform()
        
        # Load metadata
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by split
        split_map = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_map[split]]
        self.metadata = self.metadata.reset_index(drop=True)
        
    def _default_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, row['img_filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        label = int(row['y'])  # Bird type: 0=landbird, 1=waterbird
        group = int(row['group'])  # 0-3, combination of bird type and background
        
        return image, label, group
    
    def get_group_counts(self):
        """Return count of samples in each group."""
        return self.metadata.groupby('group').size().to_dict()


def get_waterbirds_loaders(root_dir, batch_size=32, num_workers=4):
    """Get train/val/test dataloaders for Waterbirds."""
    train_dataset = WaterbirdsDataset(root_dir, split='train')
    val_dataset = WaterbirdsDataset(root_dir, split='val')
    test_dataset = WaterbirdsDataset(root_dir, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 5.2 ResNet with Feature Hooks (`src/models/resnet.py`)

```python
"""ResNet models with intermediate feature extraction."""
import torch
import torch.nn as nn
from torchvision import models

class ResNetWithFeatures(nn.Module):
    """
    ResNet wrapper that returns both logits and intermediate features.
    
    Args:
        arch: 'resnet18' or 'resnet50'
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
    """
    
    def __init__(self, arch='resnet50', num_classes=2, pretrained=True):
        super().__init__()
        
        # Load base model
        if arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Replace classifier
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Storage for intermediate features
        self.features = {}
        
        # Register hooks for feature extraction
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Hook into layer4 (penultimate layer)
        self.backbone.layer4.register_forward_hook(get_hook('layer4'))
        # Optional: layer3 for multi-layer distillation
        self.backbone.layer3.register_forward_hook(get_hook('layer3'))
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, 224, 224]
            return_features: If True, return (logits, features_dict)
        
        Returns:
            logits: Class predictions [B, num_classes]
            features: Dict of intermediate features (if return_features=True)
        """
        # Clear previous features
        self.features = {}
        
        # Forward through backbone
        logits = self.backbone(x)
        
        if return_features:
            # Apply global average pooling to spatial features
            pooled_features = {}
            for name, feat in self.features.items():
                # feat shape: [B, C, H, W] -> [B, C]
                pooled_features[name] = feat.mean(dim=[2, 3])
            return logits, pooled_features
        
        return logits
    
    def get_penultimate_features(self, x):
        """Get only penultimate (layer4) features."""
        _, features = self.forward(x, return_features=True)
        return features['layer4']


class FeatureAdapter(nn.Module):
    """
    Adapter to match student feature dimensions to teacher.
    
    For ResNet-18 -> ResNet-50: 512 -> 2048
    """
    
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        if student_dim != teacher_dim:
            self.adapter = nn.Linear(student_dim, teacher_dim)
        else:
            self.adapter = nn.Identity()
    
    def forward(self, x):
        return self.adapter(x)
```

### 5.3 AGRE-KD Loss (`src/losses/agre_kd.py`)

```python
"""AGRE-KD: Adaptive Group Robust Ensemble Knowledge Distillation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AGREKDLoss(nn.Module):
    """
    AGRE-KD loss with gradient-based teacher weighting.
    
    Args:
        temperature: Softmax temperature for KD
        alpha: Weight for KD loss vs classification loss (1.0 = pure KD)
        gamma: Weight for feature distillation loss
    """
    
    def __init__(self, temperature=4.0, alpha=1.0, gamma=0.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_teacher_weights(self, student, teachers, biased_model, 
                                 images, criterion='kl'):
        """
        Compute per-sample weights for each teacher based on gradient alignment
        with the biased model.
        
        Args:
            student: Student model
            teachers: List of teacher models
            biased_model: Biased reference model (ERM-trained)
            images: Input batch
            criterion: 'kl' for KL divergence or 'ce' for cross-entropy
        
        Returns:
            weights: Tensor of shape [num_teachers, batch_size]
        """
        batch_size = images.size(0)
        num_teachers = len(teachers)
        
        # Get student logits
        student_logits = student(images)
        
        # Compute gradient w.r.t. biased model
        biased_logits = biased_model(images)
        biased_loss = self._kd_loss(student_logits, biased_logits)
        
        # Get gradient direction for biased model
        student.zero_grad()
        biased_loss.backward(retain_graph=True)
        biased_grad = self._get_gradient_vector(student)
        biased_grad = F.normalize(biased_grad, dim=0)
        
        # Compute weights for each teacher
        weights = torch.zeros(num_teachers, batch_size, device=images.device)
        
        for t_idx, teacher in enumerate(teachers):
            teacher_logits = teacher(images)
            teacher_loss = self._kd_loss(student_logits, teacher_logits)
            
            student.zero_grad()
            teacher_loss.backward(retain_graph=True)
            teacher_grad = self._get_gradient_vector(student)
            teacher_grad = F.normalize(teacher_grad, dim=0)
            
            # Weight = 1 - cosine_similarity(teacher_grad, biased_grad)
            # Higher weight for teachers that disagree with biased model
            cos_sim = torch.sum(teacher_grad * biased_grad)
            weights[t_idx] = 1 - cos_sim
        
        # Normalize weights per sample
        weights = F.softmax(weights, dim=0)
        
        return weights
    
    def _kd_loss(self, student_logits, teacher_logits):
        """Compute KL divergence loss for knowledge distillation."""
        T = self.temperature
        student_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits.detach() / T, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
    
    def _get_gradient_vector(self, model):
        """Flatten all gradients into a single vector."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads)
    
    def forward(self, student_logits, teacher_logits_list, labels, 
                student_features=None, teacher_features=None, weights=None):
        """
        Compute combined loss.
        
        Args:
            student_logits: Student predictions [B, C]
            teacher_logits_list: List of teacher predictions, each [B, C]
            labels: Ground truth labels [B]
            student_features: Student penultimate features [B, D_s] (optional)
            teacher_features: Teacher penultimate features [B, D_t] (optional)
            weights: Pre-computed teacher weights [num_teachers, B] (optional)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # 1. Classification loss (if alpha < 1)
        if self.alpha < 1.0:
            ce_loss = F.cross_entropy(student_logits, labels)
            loss_dict['ce'] = ce_loss.item()
        else:
            ce_loss = 0.0
            loss_dict['ce'] = 0.0
        
        # 2. Weighted KD loss
        if weights is None:
            # Simple average if no weights provided
            weights = torch.ones(len(teacher_logits_list), student_logits.size(0))
            weights = weights / len(teacher_logits_list)
            weights = weights.to(student_logits.device)
        
        kd_loss = 0.0
        for t_idx, teacher_logits in enumerate(teacher_logits_list):
            t_loss = self._kd_loss(student_logits, teacher_logits)
            # Weight by teacher importance (averaged across batch for simplicity)
            kd_loss += weights[t_idx].mean() * t_loss
        
        loss_dict['kd'] = kd_loss.item()
        
        # 3. Feature distillation loss (if gamma > 0)
        if self.gamma > 0 and student_features is not None and teacher_features is not None:
            feat_loss = F.mse_loss(student_features, teacher_features.detach())
            loss_dict['feat'] = feat_loss.item()
        else:
            feat_loss = 0.0
            loss_dict['feat'] = 0.0
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss + self.gamma * feat_loss
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
```

### 5.4 Metrics (`src/utils/metrics.py`)

```python
"""Evaluation metrics for group robustness."""
import torch
import numpy as np
from collections import defaultdict

def compute_accuracy(model, dataloader, device='cuda'):
    """Compute overall accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, groups in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def compute_group_accuracies(model, dataloader, num_groups=4, device='cuda'):
    """
    Compute per-group accuracies.
    
    Returns:
        group_accs: Dict mapping group_id -> accuracy
        wga: Worst-group accuracy
        avg_acc: Average accuracy
    """
    model.eval()
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels, groups in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                g = groups[i].item()
                group_total[g] += 1
                if predicted[i] == labels[i]:
                    group_correct[g] += 1
    
    # Compute per-group accuracy
    group_accs = {}
    for g in range(num_groups):
        if group_total[g] > 0:
            group_accs[g] = group_correct[g] / group_total[g]
        else:
            group_accs[g] = 0.0
    
    # Worst-group accuracy
    wga = min(group_accs.values())
    
    # Average accuracy (weighted by group size)
    total_correct = sum(group_correct.values())
    total_samples = sum(group_total.values())
    avg_acc = total_correct / total_samples
    
    return group_accs, wga, avg_acc

def print_results(group_accs, wga, avg_acc, group_names=None):
    """Pretty print evaluation results."""
    if group_names is None:
        group_names = {
            0: "Landbird + Land",
            1: "Landbird + Water", 
            2: "Waterbird + Land",
            3: "Waterbird + Water"
        }
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\n{'Group':<25} {'Accuracy':>10}")
    print("-"*35)
    for g, acc in group_accs.items():
        name = group_names.get(g, f"Group {g}")
        print(f"{name:<25} {acc*100:>9.2f}%")
    print("-"*35)
    print(f"{'Worst-Group Accuracy':<25} {wga*100:>9.2f}%")
    print(f"{'Average Accuracy':<25} {avg_acc*100:>9.2f}%")
    print("="*50 + "\n")
```

---

## 6. Training Pipeline

### 6.1 Teacher Training Notebook (`notebooks/01_train_teachers.ipynb`)

```python
# Cell 1: Setup (use the setup cell from Section 3.1)

# Cell 2: Configuration
CONFIG = {
    'dataset': 'waterbirds',
    'data_path': f'{DATA_PATH}/waterbirds',
    'arch': 'resnet50',
    'num_classes': 2,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'num_teachers': 3,
    'seed_base': 42,
}

# Cell 3: Training loop
from src.data.waterbirds import get_waterbirds_loaders
from src.models.resnet import ResNetWithFeatures
from src.utils.metrics import compute_group_accuracies, print_results
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_teacher(config, seed):
    """Train a single teacher model."""
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Data
    train_loader, val_loader, test_loader = get_waterbirds_loaders(
        config['data_path'], batch_size=config['batch_size']
    )
    
    # Model
    model = ResNetWithFeatures(
        arch=config['arch'], 
        num_classes=config['num_classes'],
        pretrained=True
    ).cuda()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_wga = 0
    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for images, labels, groups in pbar:
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Evaluate
        group_accs, wga, avg_acc = compute_group_accuracies(model, val_loader)
        print(f"Epoch {epoch+1}: Avg={avg_acc*100:.2f}%, WGA={wga*100:.2f}%")
        
        # Save best
        if wga > best_wga:
            best_wga = wga
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wga': wga,
                'avg_acc': avg_acc,
            }, f"{CHECKPOINT_PATH}/teacher_seed{seed}_best.pt")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{CHECKPOINT_PATH}/teacher_seed{seed}_epoch{epoch+1}.pt")
    
    return model, best_wga

# Cell 4: Train all teachers
results = {}
for i in range(CONFIG['num_teachers']):
    seed = CONFIG['seed_base'] + i
    print(f"\n{'='*50}")
    print(f"Training Teacher {i+1} with seed {seed}")
    print(f"{'='*50}\n")
    
    model, wga = train_teacher(CONFIG, seed)
    results[f'teacher_{i+1}'] = {'seed': seed, 'wga': wga}

print("\nTeacher Training Complete!")
for name, res in results.items():
    print(f"{name}: WGA = {res['wga']*100:.2f}%")
```

### 6.2 Student Distillation Notebook (`notebooks/03_train_student.ipynb`)

```python
# Cell 1: Setup

# Cell 2: Load teachers
def load_teachers(checkpoint_path, num_teachers=3, seed_base=42):
    """Load trained teacher models."""
    teachers = []
    for i in range(num_teachers):
        seed = seed_base + i
        model = ResNetWithFeatures(arch='resnet50', num_classes=2, pretrained=False).cuda()
        checkpoint = torch.load(f"{checkpoint_path}/teacher_seed{seed}_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        teachers.append(model)
        print(f"Loaded teacher {i+1} (seed {seed}): WGA = {checkpoint['wga']*100:.2f}%")
    return teachers

teachers = load_teachers(CHECKPOINT_PATH)

# Select one as biased model (first ERM teacher)
biased_model = teachers[0]

# Cell 3: Experiment configurations
EXPERIMENTS = {
    'baseline_agre_kd': {'alpha': 1.0, 'gamma': 0.0, 'name': 'AGRE-KD (baseline)'},
    'exp1_alpha07': {'alpha': 0.7, 'gamma': 0.0, 'name': 'Exp1: α=0.7'},
    'exp1_alpha09': {'alpha': 0.9, 'gamma': 0.0, 'name': 'Exp1: α=0.9'},
    'exp2_gamma01': {'alpha': 1.0, 'gamma': 0.1, 'name': 'Exp2: γ=0.1'},
    'exp2_gamma025': {'alpha': 1.0, 'gamma': 0.25, 'name': 'Exp2: γ=0.25'},
    'exp3_combined': {'alpha': 0.7, 'gamma': 0.1, 'name': 'Exp3: α=0.7, γ=0.1'},
}

# Cell 4: Training function
def train_student(teachers, biased_model, config, exp_name):
    """Train student with specified configuration."""
    from src.losses.agre_kd import AGREKDLoss
    from src.models.resnet import ResNetWithFeatures, FeatureAdapter
    
    # Student model
    student = ResNetWithFeatures(arch='resnet18', num_classes=2, pretrained=True).cuda()
    
    # Feature adapter (if using feature distillation)
    adapter = FeatureAdapter(512, 2048).cuda() if config['gamma'] > 0 else None
    
    # Loss
    criterion = AGREKDLoss(
        temperature=4.0,
        alpha=config['alpha'],
        gamma=config['gamma']
    )
    
    # Optimizer (include adapter params if used)
    params = list(student.parameters())
    if adapter:
        params += list(adapter.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    
    # Data
    train_loader, val_loader, test_loader = get_waterbirds_loaders(
        f'{DATA_PATH}/waterbirds', batch_size=32
    )
    
    best_wga = 0
    history = {'train_loss': [], 'val_wga': [], 'val_avg': []}
    
    for epoch in range(30):
        student.train()
        epoch_loss = 0
        
        for images, labels, groups in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.cuda(), labels.cuda()
            
            # Get student outputs
            student_logits, student_feats = student(images, return_features=True)
            student_penultimate = student_feats['layer4']
            
            # Get teacher outputs
            teacher_logits_list = []
            teacher_penultimate = None
            
            with torch.no_grad():
                for t in teachers:
                    t_logits, t_feats = t(images, return_features=True)
                    teacher_logits_list.append(t_logits)
                
                # Average teacher features for feature distillation
                if config['gamma'] > 0:
                    teacher_feats_list = [t(images, return_features=True)[1]['layer4'] for t in teachers]
                    teacher_penultimate = torch.stack(teacher_feats_list).mean(0)
            
            # Adapt student features
            if adapter and config['gamma'] > 0:
                student_penultimate = adapter(student_penultimate)
            
            # Compute loss
            optimizer.zero_grad()
            loss, loss_dict = criterion(
                student_logits, 
                teacher_logits_list,
                labels,
                student_penultimate if config['gamma'] > 0 else None,
                teacher_penultimate
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        group_accs, wga, avg_acc = compute_group_accuracies(student, val_loader)
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_wga'].append(wga)
        history['val_avg'].append(avg_acc)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
              f"Avg={avg_acc*100:.2f}%, WGA={wga*100:.2f}%")
        
        # Save best
        if wga > best_wga:
            best_wga = wga
            torch.save({
                'model_state_dict': student.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'config': config,
                'wga': wga,
                'avg_acc': avg_acc,
                'group_accs': group_accs,
            }, f"{CHECKPOINT_PATH}/student_{exp_name}_best.pt")
    
    # Final test evaluation
    checkpoint = torch.load(f"{CHECKPOINT_PATH}/student_{exp_name}_best.pt")
    student.load_state_dict(checkpoint['model_state_dict'])
    group_accs, wga, avg_acc = compute_group_accuracies(student, test_loader)
    
    return {
        'wga': wga,
        'avg_acc': avg_acc,
        'group_accs': group_accs,
        'history': history
    }

# Cell 5: Run all experiments
all_results = {}
for exp_name, config in EXPERIMENTS.items():
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}\n")
    
    results = train_student(teachers, biased_model, config, exp_name)
    all_results[exp_name] = results
    
    print(f"\nFinal Test Results for {config['name']}:")
    print_results(results['group_accs'], results['wga'], results['avg_acc'])

# Cell 6: Save all results
import json
with open(f"{LOG_PATH}/experiment_results.json", 'w') as f:
    # Convert numpy types for JSON serialization
    serializable = {k: {
        'wga': float(v['wga']),
        'avg_acc': float(v['avg_acc']),
        'group_accs': {str(g): float(a) for g, a in v['group_accs'].items()}
    } for k, v in all_results.items()}
    json.dump(serializable, f, indent=2)

print(f"Results saved to {LOG_PATH}/experiment_results.json")
```

---

## 7. Checkpointing Strategy

### 7.1 What to Save

```python
# For teachers
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'wga': wga,
    'avg_acc': avg_acc,
    'config': config,
}, checkpoint_path)

# For students (with adapter)
torch.save({
    'epoch': epoch,
    'model_state_dict': student.state_dict(),
    'adapter_state_dict': adapter.state_dict() if adapter else None,
    'optimizer_state_dict': optimizer.state_dict(),
    'config': exp_config,
    'wga': wga,
    'avg_acc': avg_acc,
    'group_accs': group_accs,
    'history': history,
}, checkpoint_path)
```

### 7.2 Resuming from Checkpoint

```python
def resume_training(checkpoint_path, model, optimizer, scheduler=None):
    """Resume training from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch
```

### 7.3 Colab Disconnect Recovery

Add this to your training loop:

```python
import os

# Check for existing checkpoint
checkpoint_path = f"{CHECKPOINT_PATH}/teacher_seed{seed}_latest.pt"
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint, resuming...")
    start_epoch = resume_training(checkpoint_path, model, optimizer, scheduler)

# In training loop
for epoch in range(start_epoch, config['epochs']):
    # ... training code ...
    
    # Save checkpoint every epoch (overwrites previous)
    torch.save({...}, checkpoint_path)
```

---

## 8. Team Coordination

### 8.1 Communication Plan

| Time | Check-in |
|------|----------|
| Morning | Share what you'll work on today |
| Evening | Share what you completed, any blockers |
| Ad-hoc | Message when experiments finish |

### 8.2 Shared Google Drive Structure

```
MyDrive/
└── robust-ensemble-kd/
    ├── checkpoints/
    │   ├── teacher_seed42_best.pt      # Person A
    │   ├── teacher_seed43_best.pt      # Person B
    │   ├── teacher_seed44_best.pt      # Person B
    │   ├── student_baseline_best.pt
    │   └── ...
    ├── data/
    │   └── waterbirds/                 # Downloaded once, shared
    ├── logs/
    │   ├── experiment_results.json
    │   └── training_logs/
    └── figures/
        └── (generated visualizations)
```

### 8.3 Git Workflow

```bash
# Before starting work
git pull

# After completing a feature
git add .
git commit -m "Implement feature X"
git push

# If conflicts
git stash
git pull
git stash pop
# Resolve conflicts manually
```

---

## 9. Debugging and Common Issues

### 9.1 Colab Session Timeout

**Problem**: Colab disconnects after idle timeout (90 min) or max runtime (12h Pro)

**Solutions**:
- Checkpoint frequently to Google Drive
- Use a keepalive script (but don't abuse):
```javascript
// Run in browser console (use sparingly)
function KeepClicking(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

### 9.2 Out of Memory (OOM)

**Problem**: CUDA out of memory on T4 (16GB)

**Solutions**:
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Clear cache between experiments
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint
```

### 9.3 Slow Data Loading

**Problem**: Training bottlenecked by data loading

**Solutions**:
```python
# Increase workers
DataLoader(..., num_workers=4, pin_memory=True)

# Pre-load data to RAM (if small enough)
# Or copy to Colab's local SSD:
!cp -r {DATA_PATH}/waterbirds /content/waterbirds_local
```

### 9.4 Git Push Authentication

**Problem**: Can't push to GitHub from Colab

**Solution**:
```python
# Generate Personal Access Token on GitHub
# Settings > Developer Settings > Personal Access Tokens > Generate

# Set remote URL with token
!git remote set-url origin https://YOUR_TOKEN@github.com/username/repo.git
!git push
```

---

## Quick Reference: File Locations

| What | Where |
|------|-------|
| Code | GitHub repo → cloned to `/content/repo` |
| Checkpoints | Google Drive → `/content/drive/MyDrive/project/checkpoints` |
| Data | Google Drive → `/content/drive/MyDrive/project/data` |
| Logs | Google Drive → `/content/drive/MyDrive/project/logs` |
| Temp files | Colab local → `/content/` (lost on disconnect) |

---

## Checklist Before Starting

- [ ] Create GitHub repository
- [ ] Create Google Drive folder structure
- [ ] Download Waterbirds to Google Drive
- [ ] Share Drive folder with teammate
- [ ] Test setup cell works in Colab
- [ ] Verify GPU is T4 in Colab
- [ ] Set up git authentication
