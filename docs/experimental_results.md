# Experimental Results: Feature Distillation in Group-Robust Ensemble Knowledge Distillation

**Project**: 6.7960 Deep Learning, Fall 2025
**Date**: December 8, 2025

---

## 1. Introduction and Motivation

### 1.1 Background: AGRE-KD

This work extends **AGRE-KD** (Adaptive Group Robust Ensemble Knowledge Distillation) by Kenfack et al. (TMLR 2025), which addresses bias amplification in ensemble knowledge distillation.

**The Problem**: When distilling knowledge from teacher ensembles to a student, traditional methods (simple averaging) can amplify spurious correlations, hurting worst-group accuracy (WGA). Even with DFR-debiased teachers achieving 90%+ WGA, the distilled student often performs significantly worse.

**AGRE-KD's Solution**: Use gradient-based teacher weighting. For each sample, compute:

$$W_t(x_i) = 1 - \langle \nabla \ell_i^t(\theta), \nabla \ell_i^b(\theta) \rangle$$

Teachers whose gradients align with a biased reference model are downweighted, ensuring the student learns from teachers that disagree with bias.

### 1.2 The Gap We Address

The AGRE-KD paper explicitly states (Section 2):

> *"We restrict ourselves to logit distillation and leave feature distillation for future exploration."*

This project directly investigates that open question: **Does adding intermediate feature distillation improve group robustness in AGRE-KD?**

### 1.3 Research Questions

1. **Feature Distillation (γ > 0)**: Does matching penultimate layer representations improve WGA?
2. **Class Labels (α < 1)**: Does adding ground-truth supervision help when teachers are debiased?
3. **Combined Approach**: Do these extensions synergize?

### 1.4 Why Feature Distillation Might Help (Hypothesis)

- Kirichenko et al. (2022) showed core features remain decodable at the penultimate layer even in biased models
- DFR operates at this layer—distilling here might transfer the "debiasing signal"
- Soft logits alone may not fully capture the representation structure teachers learned

### 1.5 Why Feature Distillation Might NOT Help (Counter-hypothesis)

- DFR only retrains the classifier head; **backbone features remain biased**
- All teachers share the same ImageNet-pretrained backbone
- Distilling biased features could reinforce spurious correlations

---

## 2. Experimental Setup

### 2.1 Dataset
- **Waterbirds**: 4,795 training images, 2 classes × 2 backgrounds = 4 groups
- **Metric**: Worst-Group Accuracy (WGA) - minimum accuracy across all subgroups

### 2.2 Architecture
- **Teachers**: 5× ResNet-50 (ImageNet pretrained, DFR-debiased)
- **Student**: ResNet-18 (ImageNet pretrained)
- **Teacher WGA**: 91.9% - 93.8% after DFR debiasing

### 2.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.001 (cosine decay) |
| Batch size | 128 |
| Epochs | 30 |
| Temperature (τ) | 4.0 |
| Seeds | 42, 43, 44 (+ original unseeded "og") |

### 2.4 Loss Function

$$\mathcal{L}_{total} = (1-\alpha) \mathcal{L}_{cls} + \alpha \mathcal{L}_{wKD} + \gamma \mathcal{L}_{feat}$$

Where:
- α controls class label weight (α=1 means no class labels)
- γ controls feature distillation weight
- $\mathcal{L}_{wKD}$ is AGRE-KD's gradient-weighted ensemble KD loss

---

## 3. Results Summary

### 3.1 Main Results (Ranked by WGA)

| Rank | Experiment | Method | α | γ | WGA (%) | n |
|------|------------|--------|---|---|---------|---|
| 1 | gamma_050 | AGRE-KD | 1.0 | 0.50 | **85.57 ± 0.36** | 3 |
| 2 | disagree_weight | Disagree-Weight | 1.0 | 0.50 | 85.31 ± 0.39 | 3 |
| 3 | exp2_gamma025 | AGRE-KD | 1.0 | 0.25 | 85.20 ± 0.97 | 4 |
| 4 | ml_L3_L4 | Multi-Layer | 1.0 | 0.50 | 85.20 | 1 |
| 5 | gamma_100 | AGRE-KD | 1.0 | 1.00 | 85.10 ± 1.03 | 3 |
| 6 | gamma_075 | AGRE-KD | 1.0 | 0.75 | 84.89 ± 0.56 | 3 |
| 7 | aver_gamma_050 | AVER | 1.0 | 0.50 | 84.89 | 1 |
| 8 | ml_L2_L3_L4 | Multi-Layer | 1.0 | 0.50 | 84.74 | 1 |
| 9 | **baseline_agrekd** | AGRE-KD | 1.0 | 0.00 | 84.62 ± 1.06 | 4 |
| 10 | aver_gamma025 | AVER | 1.0 | 0.25 | 84.42 | 1 |
| 11 | aver_gamma_100 | AVER | 1.0 | 1.00 | 84.27 | 1 |
| 12 | exp3_a09_g025 | AGRE-KD | 0.9 | 0.25 | 84.16 ± 1.01 | 3 |
| 13 | gamma_005 | AGRE-KD | 1.0 | 0.05 | 83.96 | 1 |
| 14 | aver_baseline | AVER | 1.0 | 0.00 | 83.95 ± 0.95 | 4 |

### 3.2 Key Comparisons

| Comparison | Result | Δ WGA |
|------------|--------|-------|
| AGRE-KD vs AVER (baseline) | AGRE-KD wins | +0.67% |
| Best feature dist vs baseline | γ=0.5 wins | +0.95% |
| Combined (α<1) vs baseline | Baseline wins | -0.46% |

---

## 4. Hypothesis Testing

### H1: Class Labels (α < 1) Improve WGA
**Result: FALSE**

| Experiment | α | γ | WGA |
|------------|---|---|-----|
| baseline_agrekd | 1.0 | 0.0 | 84.62% |
| exp3_a09_g025 | 0.9 | 0.25 | 84.16% |

Adding class labels (α=0.9) **decreased** WGA by 0.46%. This contradicts the AGRE-KD paper's suggestion that class labels could help when teachers are biased.

**Possible explanation**: Our DFR-debiased teachers have high WGA (91-94%), so the class label signal may interfere with the already-good teacher guidance rather than correct for bias.

### H2: Feature Distillation (γ > 0) Improves WGA
**Result: TRUE (marginal)**

| γ | WGA | Δ vs Baseline |
|---|-----|---------------|
| 0.00 | 84.62% | — |
| 0.05 | 83.96% | -0.66% |
| 0.25 | 85.20% | +0.58% |
| 0.50 | **85.57%** | **+0.95%** |
| 0.75 | 84.89% | +0.27% |
| 1.00 | 85.10% | +0.48% |

Feature distillation provides a **marginal improvement** of ~1% at optimal γ=0.5. However:
- The improvement (+0.95%) is modest
- Error bars overlap significantly (baseline: ±1.06%, γ=0.5: ±0.36%)
- Not statistically significant with n=3-4 seeds

### H3: Combined Approach is Best
**Result: FALSE**

The combined approach (α=0.9, γ=0.25) at 84.16% performed **worse** than:
- Baseline (84.62%)
- Feature-only γ=0.25 (85.20%)
- Feature-only γ=0.5 (85.57%)

Class labels appear to hurt rather than help, negating any benefit from feature distillation.

---

## 5. Gamma Sensitivity Analysis

### 5.1 AGRE-KD Gamma Curve

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
1. Very low γ (0.05) hurts performance
2. Optimal γ appears to be around 0.5
3. Diminishing returns beyond γ=0.5
4. All γ > 0.1 outperform baseline

### 5.2 Seed Variance

| γ | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---|---------|---------|---------|------------|
| 0.00 | 84.58% | 85.67% | 83.18% | 84.48 ± 1.02% |
| 0.25 | 84.11% | 85.67% | 84.74% | 84.84 ± 0.64% |
| 0.50 | 85.36% | 85.98% | 85.36% | 85.57 ± 0.29% |
| 0.75 | 84.42% | 84.74% | 85.51% | 84.89 ± 0.46% |
| 1.00 | 85.36% | 83.96% | 85.98% | 85.10 ± 0.84% |

**Key insight**: γ=0.5 has the **tightest variance** (±0.29%), suggesting it's the most stable configuration.

---

## 6. Exploratory Methods

Beyond varying γ, we explored alternative feature aggregation strategies to test whether more sophisticated approaches could improve group robustness.

### 6.1 Disagree-Weight Feature Distillation

**Motivation**: If teachers disagree on a sample's features, that disagreement might signal that the sample is "hard" or from a minority group. Weighting features by disagreement could emphasize these informative samples.

**Method**: For each sample, compute teacher feature variance (disagreement) and use it to weight the feature loss:

$$w_i = \text{Var}(\{f_t(x_i)\}_{t=1}^T)$$

Samples where teachers disagree more receive higher weight in the feature distillation loss.

**Result**: 85.31% WGA—competitive with standard AGRE-KD (85.57%) but slightly worse. The disagreement signal doesn't provide additional benefit beyond gradient-based weighting.

### 6.2 Multi-Layer Feature Distillation

**Motivation**: DFR operates at the penultimate layer, but earlier layers might contain complementary information. Distilling from multiple layers could capture richer representations.

**Method**: Extract features from ResNet layers 3 and 4 (and combinations), using adapters to match student-teacher dimensions:

- **L3+L4**: Distill from both layer3 and layer4
- **L2+L3+L4**: Distill from layers 2, 3, and 4

**Results**:
| Configuration | WGA |
|---------------|-----|
| L3+L4 | 85.20% |
| L2+L3+L4 | 84.74% |
| Single layer (L4) | 85.57% |

Multi-layer distillation doesn't improve results. This aligns with our hypothesis: since DFR only debiases the classifier head, earlier backbone layers remain equally biased. Adding more biased layers doesn't help.

### 6.3 Method Comparison Summary (γ=0.5)

| Method | WGA | Description |
|--------|-----|-------------|
| AGRE-KD | 85.57% | Gradient-weighted teacher ensemble |
| Disagree-Weight | 85.31% | Weight features by teacher agreement |
| Multi-Layer (L3+L4) | 85.20% | Distill from layers 3 and 4 |
| AVER | 84.89% | Simple teacher averaging |

**Key Findings**:
1. AGRE-KD gradient weighting provides the best results
2. Disagree-weighting is competitive but doesn't beat gradient-based weighting
3. Multi-layer distillation doesn't improve over single-layer (backbone remains biased)
4. All methods with feature distillation outperform their baselines

---

## 7. AGRE-KD vs AVER

### 7.1 Baseline Comparison (γ=0)

| Method | WGA | n |
|--------|-----|---|
| AGRE-KD | 84.62 ± 1.06% | 4 |
| AVER | 83.95 ± 0.95% | 4 |
| **Difference** | **+0.67%** | |

AGRE-KD's gradient-based weighting provides a small but consistent advantage over simple averaging.

### 7.2 With Feature Distillation

| γ | AGRE-KD | AVER | Δ |
|---|---------|------|---|
| 0.00 | 84.62% | 83.95% | +0.67% |
| 0.25 | 85.20% | 84.42% | +0.78% |
| 0.50 | 85.57% | 84.89% | +0.68% |
| 1.00 | 85.10% | 84.27% | +0.83% |

The AGRE-KD advantage is **consistent across all γ values** (~0.7-0.8%).

---

## 8. Discussion

### 8.1 Why Feature Distillation Provides Limited Benefit

Our results show feature distillation provides only marginal improvement (+0.95% at best). This is likely because:

1. **DFR only debiases the classifier head**: The backbone features remain biased. Feature distillation transfers these biased features to the student.

2. **Teachers share the same backbone**: All 5 teachers use the same ImageNet-pretrained ResNet-50 backbone. DFR only retrains the last layer, so teacher features are nearly identical.

3. **Averaging identical features**: When teachers have similar features, the weighted average provides little additional signal beyond what a single teacher would provide.

### 8.2 Why Class Labels Hurt

Adding class labels (α < 1) decreased WGA. Possible reasons:

1. **Teacher quality is already high**: With 91-94% WGA teachers, the class label signal may conflict with the teacher's correct guidance.

2. **Class labels don't address group structure**: The cross-entropy loss treats all samples equally, potentially overriding the gradient-based weighting that helps minority groups.

### 8.3 Comparison to Paper Results

Our AGRE-KD baseline achieves **84.62%** WGA, lower than the paper's reported ~87.9%. Potential causes:
- We use 5 teachers vs paper's 10
- Different hyperparameters (lr, epochs)
- Random variance

See `docs/limitations.md` for full discussion.

---

## 9. Conclusions

### 9.1 Summary of Findings

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Class labels (α<1) help | **FALSE** | 84.16% vs 84.62% baseline |
| H2: Feature distillation (γ>0) helps | **TRUE (marginal)** | 85.57% vs 84.62% baseline |
| H3: Combined approach is best | **FALSE** | Combined worst among tested |

### 9.2 Practical Recommendations

1. **Use γ=0.5 for feature distillation** - optimal balance, lowest variance
2. **Keep α=1** - class labels don't help with high-quality debiased teachers
3. **AGRE-KD > AVER** - gradient weighting provides consistent ~0.7% improvement
4. **Single-layer distillation is sufficient** - multi-layer doesn't improve results

### 9.3 Contribution

This work provides empirical evidence that:
- Feature distillation offers limited benefit for group-robust KD when teachers are DFR-debiased
- The benefit is marginal because DFR only debiases the classifier, not backbone features
- Class labels can interfere with gradient-weighted distillation from high-quality teachers

This constitutes a valid **"negative result"** that explains why the AGRE-KD authors left feature distillation for future work - the gains are modest when using DFR-debiased teachers.

---

## 10. Future Work

1. **Test with biased teachers**: Feature distillation might help more when teachers are biased (not DFR-debiased)
2. **End-to-end debiasing**: Train teachers with group-robust methods that debias the full network, not just the classifier
3. **Larger teacher ensemble**: Test with 10 teachers to match paper's setup
4. **Different datasets**: Validate on CelebA and other spurious correlation benchmarks

---

## Appendix: Complete Experiment Log

Total experiments: 34 runs across 14 configurations

### By Category
- Baseline (AGRE-KD): 4 runs
- Baseline (AVER): 4 runs
- Feature Distillation (AGRE-KD): 17 runs
- Feature Distillation (AVER): 3 runs
- Combined (α<1): 3 runs
- Disagree Weighting: 3 runs
- Multi-Layer: 2 runs

### Seeds Used
- Primary: 42, 43, 44
- Original (unseeded): "og"

### Compute
- Platform: Google Colab (T4 GPU)
- Time per run: ~45-60 minutes
- Total compute: ~25-30 GPU hours
