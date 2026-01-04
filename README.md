# Feature Distillation for Group-Robust Knowledge Distillation

*By Dat Tran and Priscilla Leang*

**[Read the full report →](blog/index.html)**

## Overview

This project extends [AGRE-KD](https://arxiv.org/abs/2306.17193) (Adaptive Group Robust Ensemble Knowledge Distillation) by investigating **feature distillation** for improving worst-group accuracy in knowledge distillation.

**Problem**: Ensemble KD can amplify spurious correlations, hurting minority group accuracy.

**AGRE-KD**: Weights teachers by gradient disagreement with a biased model—teachers that disagree with bias get upweighted.

**Our Extension**: The AGRE-KD paper leaves feature distillation for future work. We investigate whether distilling intermediate features (not just logits) improves group robustness.

## Key Findings

| Finding | Result |
|---------|--------|
| Feature distillation (γ=0.5) | +0.95% WGA (marginal improvement) |
| Class labels (α<1) | Hurts performance with high-quality teachers |
| Multi-layer distillation | Worse than single layer |
| AGRE-KD vs simple averaging | +0.7% consistent advantage |

**Why limited improvement?** DFR-debiased teachers only have debiased classifiers—backbone features remain biased. Feature distillation transfers these biased features.

## Results

| Method | WGA |
|--------|-----|
| AGRE-KD + Features (γ=0.5) | **85.57 ± 0.36%** |
| AGRE-KD Baseline | 84.62 ± 1.06% |
| AVER Baseline | 83.95 ± 0.95% |

## Setup

```bash
# Dataset: Waterbirds
# Teachers: 5x ResNet-50 (DFR-debiased, 91-94% WGA)
# Student: ResNet-18
# Training: 30 epochs, lr=0.001, batch_size=128
```

## Project Structure

```text
├── code/
│   ├── train.py          # Training loop
│   ├── losses.py         # Loss functions (KD, feature distillation)
│   ├── models.py         # Model definitions
│   └── notebooks/        # Experiment notebooks
├── blog/                 # Project report (HTML)
│   └── index.html
└── docs/
    └── experimental_results.md
```

## Citation

Based on:

```bibtex
@article{kenfack2025agrekd,
  title={AGRE-KD: Adaptive Group Robust Ensemble Knowledge Distillation},
  author={Kenfack, et al.},
  journal={TMLR},
  year={2025}
}
```
