# Feature Distillation for Group-Robust Knowledge Distillation

**Course**: 6.7960 Deep Learning, Fall 2025 | **Date**: December 8, 2025

---

## Overview

**Problem**: When distilling from teacher ensembles, students can learn spurious correlations (e.g., "water background → waterbird") that hurt minority group accuracy.

**AGRE-KD Solution**: Weight teachers by gradient disagreement with a biased reference model. Teachers that disagree with bias get upweighted.

**Our Extension**: The AGRE-KD paper states *"We leave feature distillation for future exploration."* We investigate whether distilling intermediate features (not just logits) improves group robustness.

**Setup**: 5 DFR-debiased ResNet-50 teachers (91-94% WGA) → ResNet-18 student, Waterbirds dataset, 30 epochs.

---

## Loss Function and Training Objective

### Complete Loss

$$\mathcal{L}_{total} = (1-\alpha)\mathcal{L}_{cls} + \alpha\mathcal{L}_{wKD} + \gamma\mathcal{L}_{feat}$$

### Component Breakdown

| Term | Formula | Role |
|------|---------|------|
| $\mathcal{L}_{cls}$ | $\text{CE}(y, \hat{y}_s)$ | Cross-entropy with ground-truth labels |
| $\mathcal{L}_{wKD}$ | $\text{KL}(\sigma(z_s/\tau) \| \sigma(\bar{z}_T/\tau))$ | KL divergence between student and weighted teacher logits |
| $\mathcal{L}_{feat}$ | $\|f_s - \bar{f}_T\|_2^2$ | MSE between student and averaged teacher features |

### Hyperparameters

| Param | Range | Meaning |
|-------|-------|---------|
| α | [0, 1] | Weight on KD vs class labels. α=1 means pure KD (no labels) |
| γ | [0, ∞) | Weight on feature distillation. γ=0 means logits only |
| τ | 4.0 | Temperature for softening logits. Higher τ = softer distributions |

### AGRE-KD Teacher Weighting

The key innovation is how teacher logits are combined. Instead of simple averaging:

$$\bar{z}_T = \frac{1}{T}\sum_{t=1}^{T} z_t \quad \text{(AVER)}$$

AGRE-KD computes per-sample, per-teacher weights based on gradient alignment with a biased model:

$$W_t(x_i) = \text{ReLU}\left(1 - \cos(\nabla\ell_i^t, \nabla\ell_i^b)\right)$$

$$\bar{z}_T(x_i) = \frac{\sum_{t=1}^{T} W_t(x_i) \cdot z_t(x_i)}{\sum_{t=1}^{T} W_t(x_i)} \quad \text{(AGRE-KD)}$$

Where:
- $\nabla\ell_i^t$ = gradient from teacher t's prediction on sample i
- $\nabla\ell_i^b$ = gradient from biased model's prediction on sample i
- $\cos(\cdot, \cdot)$ = cosine similarity

**Intuition**: If teacher gradient aligns with biased model gradient → teacher is giving biased guidance → downweight. If they oppose → teacher disagrees with bias → upweight.

### Feature Aggregation

For feature distillation, we average teacher features (optionally with AGRE weights):

$$\bar{f}_T = \frac{1}{T}\sum_{t=1}^{T} f_t \quad \text{(standard)}$$

Since teacher ResNet-50 (2048-dim) and student ResNet-18 (512-dim) have different feature dimensions, we use a projection layer:

$$\mathcal{L}_{feat} = \|W_{proj} \cdot f_s - \bar{f}_T\|_2^2$$

Where $W_{proj}$ is a learned 512→2048 linear projection.

---

## Main Results

All experiments use optimal hyperparameters (α=1.0, γ=0.5) unless otherwise noted.

| Experiment | Description | WGA | Δ vs Baseline |
|------------|-------------|-----|---------------|
| **AGRE-KD + Features** | Gradient-weighted KD + feature distillation | **85.57 ± 0.36%** | **+0.95%** |
| Disagree-Weight | Weight features by teacher variance | 85.31 ± 0.39% | +0.69% |
| Multi-Layer (L3+L4) | Distill from layers 3 and 4 | 85.20% | +0.58% |
| AVER + Features | Simple averaging + feature distillation | 84.89% | +0.27% |
| AGRE-KD Baseline | Gradient-weighted KD, no features (γ=0) | 84.62 ± 1.06% | — |
| Combined (α=0.9) | Add class labels to loss | 84.16 ± 1.01% | -0.46% |
| AVER Baseline | Simple averaging, no features | 83.95 ± 0.95% | -0.67% |

---

## Experiment Details

### Experiment 1: AGRE-KD vs Simple Averaging (Baseline)

**Question**: Does gradient-based teacher weighting help?

| Method | WGA | Description |
|--------|-----|-------------|
| AGRE-KD | 84.62 ± 1.06% | Gradient-weighted teachers |
| AVER | 83.95 ± 0.95% | Simple average |
| **Δ** | **+0.67%** | |

**Finding**: AGRE-KD's gradient weighting provides consistent improvement over simple averaging.

---

### Experiment 2: Feature Distillation

**Question**: Does matching teacher features improve WGA?

| Method | WGA | Δ vs Baseline |
|--------|-----|---------------|
| AGRE-KD + Features (γ=0.5) | **85.57 ± 0.36%** | **+0.95%** |
| AGRE-KD Baseline (γ=0) | 84.62 ± 1.06% | — |

**Finding**: Feature distillation provides **marginal improvement** (+0.95%). Also reduces variance (±0.36% vs ±1.06%), making training more stable.

**Why marginal?** DFR only debiases the classifier head—backbone features remain biased. We're distilling biased features.

---

### Experiment 3: Class Labels (α < 1)

**Question**: Does adding ground-truth supervision help?

| Method | α | γ | WGA |
|--------|---|---|-----|
| Feature only | 1.0 | 0.25 | 85.20% |
| **Combined** | **0.9** | **0.25** | **84.16%** |

**Finding**: Class labels **hurt** performance (-1.04% vs feature-only). With high-quality teachers (91-94% WGA), the label signal interferes with gradient weighting.

---

### Experiment 4: Disagree-Weight Features

**Question**: Can we weight features by teacher disagreement (like AGRE-KD does for logits)?

**Method**: Weight feature loss by variance across teachers: $w_i = \text{Var}(\{f_t(x_i)\})$

| Method | WGA |
|--------|-----|
| Standard AGRE-KD + Features | 85.57% |
| Disagree-Weight Features | 85.31% |

**Finding**: Disagree-weighting doesn't help (-0.26%). Teachers share the same backbone, so feature disagreement is minimal and uninformative.

---

### Experiment 5: Multi-Layer Distillation

**Question**: Does distilling from earlier layers capture richer information?

**Method**: Distill from ResNet layers 2, 3, 4 using dimension-matching adapters.

| Layers | WGA |
|--------|-----|
| L4 only (standard) | 85.57% |
| L3 + L4 | 85.20% |
| L2 + L3 + L4 | 84.74% |

**Finding**: More layers = **worse** performance. Earlier layers encode low-level features (textures, backgrounds)—exactly the spurious correlations we want to avoid.

---

## Hypotheses Summary

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Class labels (α<1) help | **FALSE** | 84.16% vs 84.62% |
| H2: Feature distillation (γ>0) helps | **TRUE (marginal)** | 85.57% vs 84.62% (+0.95%) |
| H3: Combined is best | **FALSE** | Combined worst of feature configs |

---

## Key Takeaways

1. **Feature distillation: marginal gain** (+0.95%) because DFR doesn't debias backbone features
2. **Optimal config**: α=1.0, γ=0.5, single layer (layer4)
3. **Class labels hurt** with high-quality teachers
4. **Multi-layer hurts**—earlier layers are more biased
5. **AGRE-KD > AVER** consistently (+0.7%)

---

## Why Limited Improvement?

```
ResNet: [layer1 → layer2 → layer3 → layer4] → [FC]
              BIASED BACKBONE                  ↑
                                          DFR only
                                         debiases here
```

DFR retrains only the final classifier. Backbone features remain biased. Feature distillation transfers these biased features, limiting its benefit.

**This explains why AGRE-KD authors left feature distillation for future work—gains are modest with DFR-debiased teachers.**

---

## Ablation Studies

### A1: Feature Weight (γ) Sensitivity

How does varying γ affect performance?

| γ | WGA | Δ vs Baseline | Variance |
|---|-----|---------------|----------|
| 0.00 | 84.62% | — | ±1.06% |
| 0.05 | 83.96% | -0.66% | N/A |
| 0.25 | 85.20% | +0.58% | ±0.97% |
| **0.50** | **85.57%** | **+0.95%** | **±0.36%** |
| 0.75 | 84.89% | +0.27% | ±0.56% |
| 1.00 | 85.10% | +0.48% | ±1.03% |

```
WGA (%)
  86 |           * (γ=0.5)
     |       *       * (γ=0.25, γ=1.0)
  85 |           * (γ=0.75)
     |   * (baseline)
  84 |
     | * (γ=0.05)
  83 |
     +---+---+---+---+---+---
        0  0.05 0.25 0.5 0.75 1.0
                γ
```

**Observations**:
- Very low γ (0.05) hurts—feature signal too weak
- Optimal γ = 0.5 (best WGA, lowest variance)
- Diminishing returns beyond γ=0.5

---

### A2: AGRE-KD vs AVER Across γ Values

Does the AGRE-KD advantage hold across different feature weights?

| γ | AGRE-KD | AVER | Δ |
|---|---------|------|---|
| 0.00 | 84.62% | 83.95% | +0.67% |
| 0.25 | 85.20% | 84.42% | +0.78% |
| 0.50 | 85.57% | 84.89% | +0.68% |
| 1.00 | 85.10% | 84.27% | +0.83% |

**Finding**: AGRE-KD advantage is consistent (~0.7-0.8%) across all γ values. Effects are additive—gradient weighting and feature distillation provide independent benefits.

---

### A3: Seed Variance Analysis

How stable are results across random seeds (42, 43, 44)?

| Config | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| Baseline (γ=0) | 84.58% | 85.67% | 83.18% | 84.48 ± 1.02% |
| γ=0.25 | 84.11% | 85.67% | 84.74% | 84.84 ± 0.64% |
| γ=0.50 | 85.36% | 85.98% | 85.36% | 85.57 ± 0.29% |
| γ=0.75 | 84.42% | 84.74% | 85.51% | 84.89 ± 0.46% |
| γ=1.00 | 85.36% | 83.96% | 85.98% | 85.10 ± 0.84% |

**Finding**: γ=0.5 has the tightest variance (±0.29%), making it the most reliable configuration.
