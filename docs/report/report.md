# Improving Group Robustness in Ensemble Knowledge Distillation: Beyond Logit Matching

**Dat Tran** · **Priscilla Leang**

---

## Introduction

Neural networks can learn spurious correlations in the data, often leading to performance degradation for underrepresented subgroups. Studies have demonstrated that this disparity is amplified when knowledge is distilled from a complex teacher model to a simpler student model (Lukasik et al., 2023; Lee & Lee, 2023; Wang et al., 2023). Deep ensemble models can enhance both generalization and worst-case group performance (Ganaie et al., 2022; Ko et al., 2023; Kenfack et al., 2021), but evaluating multiple models at test time is computationally expensive, making them impractical for edge deployment. Ensemble knowledge distillation addresses this by distilling multiple teachers into a single student (You et al., 2017; Radwan et al., 2024).

A natural approach is to average teacher predictions, but this can degrade worst-group accuracy: the consensus favors directions that minimize average error, not worst-case error. **AGRE-KD** (Kenfack et al., 2025) addresses this through gradient-based teacher weighting: rather than equally averaging predictions, it downweights teachers whose gradients align with a biased reference model. This achieves state-of-the-art worst-group accuracy on benchmarks like Waterbirds. However, AGRE-KD distills only teacher logits—the authors explicitly note that feature distillation remains an open direction for future work.

This work investigates that direction. We extend AGRE-KD with a feature distillation term and explore whether intermediate representations can provide additional debiasing signal beyond logits alone.

Our experiments show that feature distillation provides modest but consistent improvement with notably reduced variance across trials. We also find that these gains are constrained by how teachers are debiased—specifically, the debiasing method used affects only the classifier layer while leaving backbone features unchanged. This analysis clarifies when feature distillation helps and suggests that stronger gains may require debiasing methods that operate on the full network, not just the classifier.

---

## Background & Related Work

### Spurious Correlations & Worst-Group Accuracy

Neural networks excel at finding patterns—sometimes the wrong ones. Given training data where waterbirds typically appear against water backgrounds and landbirds against land backgrounds, a model might learn "water background → waterbird" rather than actual bird features. This works on average but fails on **minority groups**: a waterbird photographed on land gets misclassified because the model relies on background rather than the bird itself (Sagawa et al., 2020).

**Worst-Group Accuracy (WGA)** measures robustness against such failures—the accuracy on the worst-performing subgroup rather than the overall average. Maximizing WGA ensures models work for all groups, not just the easy majority.

### Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) trains a student network to match the soft probability outputs of a teacher, transferring "dark knowledge" encoded in the teacher's uncertainty. The KL divergence loss with temperature τ is:

$$
\mathcal{L}_{KD} = \text{KL}(\sigma(z_s/\tau) \| \sigma(z_t/\tau))
$$

Higher temperatures produce softer distributions, revealing more information about inter-class relationships.

### Ensemble Distillation & Bias Amplification

Ensemble distillation extends KD by averaging predictions from multiple teachers, typically producing more robust students. However, when teachers share similar biases—even partially—simple averaging can reinforce these biases.

Kenfack et al. (2025) demonstrated this **bias amplification** effect: a student distilled from debiased teachers can achieve *worse* WGA than the teachers themselves. The shared biases get amplified through the averaging process.

### AGRE-KD: Gradient-Based Teacher Weighting

AGRE-KD solves bias amplification by computing per-sample, per-teacher weights based on gradient alignment with a biased reference model. The key insight: if a teacher's gradient aligns with what a biased model would predict, that teacher is likely giving biased guidance for that sample.

The weighting formula is:

$$
W_t(x_i) = \text{ReLU}\left(1 - \cos(\nabla\ell_i^t, \nabla\ell_i^b)\right)
$$

Teachers whose gradients oppose the biased model get upweighted; those that align get downweighted.

![AGRE-KD gradient weighting illustration showing how teachers aligned with the biased direction are downweighted while those that deviate are upweighted. Adapted from Kenfack et al. (2025).](../blog/images/weighting.png)

AGRE-KD uses teachers debiased via **Deep Feature Reweighting (DFR)** (Kirichenko et al., 2022). DFR makes a critical observation: even biased models learn core features—spurious correlations are primarily amplified in the final classifier layer. By simply retraining the last layer on balanced data, DFR achieves strong WGA without modifying the backbone.

However, this means **the backbone features remain biased**—only the classifier is debiased. The AGRE-KD authors note: *"We restrict ourselves to logit distillation and leave feature distillation for future exploration."*

### Our Extension: Feature Distillation

We investigate whether distilling features—not just logits—can transfer additional debiasing signal. Our full loss function is:

$$
\mathcal{L}_{total} = (1-\alpha)\mathcal{L}_{cls} + \alpha\mathcal{L}_{wKD} + \gamma\mathcal{L}_{feat}
$$

where $\mathcal{L}_{feat} = \|f_s - \bar{f}_T\|_2^2$ matches student features to the averaged teacher features.

![Our extended AGRE-KD architecture. We add a feature distillation branch that extracts penultimate layer features from teachers, averages them, and matches them to student features through a learned projection layer (2048→512 dim). The total loss combines weighted KD loss and feature MSE loss.](../blog/images/architecture.png)

We explore several design choices within this framework:

- **Feature loss weight (γ)**: Does adding feature distillation improve WGA? What weight works best?
- **Disagreement weighting**: Can we weight features by teacher disagreement, analogous to AGRE-KD's logit weighting?
- **Multi-layer distillation**: Does distilling from multiple backbone layers capture richer information?
- **Class label supervision (α < 1)**: The AGRE-KD authors show in their appendix that pure KD (α=1) outperforms mixed supervision—we confirm this in our setting.

---

## Method

### Loss Function

Our loss function combines three terms:

| Term                   | Formula                                                   | Role                                       |
| ---------------------- | --------------------------------------------------------- | ------------------------------------------ |
| $\mathcal{L}_{cls}$  | $\text{CE}(y, \hat{y}_s)$                               | Cross-entropy with ground-truth labels     |
| $\mathcal{L}_{wKD}$  | $\text{KL}(\sigma(z_s/\tau) \| \sigma(\bar{z}_T/\tau))$ | KL divergence with weighted teacher logits |
| $\mathcal{L}_{feat}$ | $\|f_s - \bar{f}_T\|_2^2$                               | MSE between student and teacher features   |

The hyperparameters control the balance:

- **α ∈ [0,1]**: Weight on KD vs class labels. α=1 means pure KD (no ground-truth labels)
- **γ ≥ 0**: Weight on feature distillation. γ=0 recovers standard AGRE-KD
- **τ = 4.0**: Temperature for softening logits

### Teacher Weighting

For logit distillation, we use AGRE-KD's gradient-based weighting. For features, we average teacher features (with optional weighting):

$$
\bar{f}_T = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

Since teacher ResNet-50 (2048-dim features) and student ResNet-18 (512-dim) have different dimensions, we learn a projection layer: $\mathcal{L}_{feat} = \|W_{proj} \cdot f_s - \bar{f}_T\|_2^2$

### Experimental Setup

- **Dataset**: Waterbirds (Sagawa et al., 2020) — 4,795 training images, 4 groups
- **Teachers**: 5 ResNet-50 models, DFR-debiased (91-94% WGA each)
- **Student**: ResNet-18, ImageNet pretrained
- **Biased model**: ResNet-50 trained with standard ERM (~73.8% WGA)
- **Training**: 30 epochs, SGD with lr=0.001, batch size 128
- **Seeds**: 42, 43, 44 for statistical significance

---

## Experiments & Results

### Experiment 1: AGRE-KD vs Simple Averaging

First, we verify that AGRE-KD's gradient weighting helps compared to simple teacher averaging (AVER).

| Method           | WGA              | n |
| ---------------- | ---------------- | - |
| AGRE-KD Baseline | 84.62 ± 1.06%   | 4 |
| AVER Baseline    | 83.95 ± 0.95%   | 4 |
| **Δ**     | **+0.67%** |   |

**Finding**: AGRE-KD provides consistent improvement over simple averaging, validating the gradient-based weighting approach.

### Experiment 2: Feature Distillation

Does adding feature distillation (γ > 0) improve WGA?

| γ              | WGA                      | Δ vs Baseline   |
| --------------- | ------------------------ | ---------------- |
| 0.00 (baseline) | 84.62 ± 1.06%           | —               |
| 0.25            | 85.20 ± 0.97%           | +0.58%           |
| **0.50**  | **85.57 ± 0.36%** | **+0.95%** |
| 0.75            | 84.89 ± 0.56%           | +0.27%           |
| 1.00            | 85.10 ± 1.03%           | +0.48%           |

**Finding**: Feature distillation provides **marginal improvement** (+0.95% at γ=0.5). Notably, γ=0.5 also has the lowest variance (±0.36% vs ±1.06%), suggesting more stable training.

### Experiment 3: Class Labels (α < 1)

Does adding ground-truth supervision help?

| α  | γ   | WGA            |
| --- | ---- | -------------- |
| 1.0 | 0.25 | 85.20%         |
| 0.9 | 0.25 | 84.16 ± 1.01% |
| 0.9 | 0.00 | 83.80%         |
| 0.7 | 0.00 | 82.55%         |

**Finding**: Class labels **hurt** performance. With high-quality teachers (91-94% WGA), the label signal interferes with gradient weighting rather than helping. Lower α consistently yields worse WGA.

### Experiment 4: Disagree-Weight Features

Can we weight features by teacher disagreement, similar to AGRE-KD's logit weighting?

| Method                      | WGA            |
| --------------------------- | -------------- |
| Standard AGRE-KD + Features | 85.57%         |
| Disagree-Weight Features    | 85.31 ± 0.39% |

**Finding**: Disagree-weighting doesn't improve results. Teachers share the same backbone, so feature disagreement is minimal and uninformative.

### Experiment 5: Multi-Layer Distillation

Does distilling from multiple layers capture richer information?

| Layers             | WGA    |
| ------------------ | ------ |
| L4 only (standard) | 85.57% |
| L3 + L4            | 85.20% |
| L2 + L3 + L4       | 84.74% |

**Finding**: More layers = **worse** performance. Earlier layers encode low-level features like textures and backgrounds—exactly the spurious correlations we want to avoid.

### Summary of Results

| Rank | Method             | α  | γ   | WGA (%)                 |
| ---- | ------------------ | --- | ---- | ----------------------- |
| 1    | AGRE-KD + Features | 1.0 | 0.50 | **85.57 ± 0.36** |
| 2    | Disagree-Weight    | 1.0 | 0.50 | 85.31 ± 0.39           |
| 3    | AGRE-KD + Features | 1.0 | 0.25 | 85.20 ± 0.97           |
| 10   | AGRE-KD Baseline   | 1.0 | 0.00 | 84.62 ± 1.06           |
| 13   | Combined (α=0.9)  | 0.9 | 0.25 | 84.16 ± 1.01           |
| 15   | AVER Baseline      | 1.0 | 0.00 | 83.95 ± 0.95           |

---

## Discussion

### Why Is Feature Distillation Only Marginally Helpful?

Our results show feature distillation provides limited benefit (+0.95%). The root cause lies in how DFR works:

```
ResNet: [layer1 → layer2 → layer3 → layer4] → [FC]
              BIASED BACKBONE                  ↑
                                          DFR only
                                         debiases here
```

DFR retrains **only the final classifier layer**. The entire backbone (layers 1-4) remains unchanged from standard biased training. When we distill features from layer4, we're distilling **biased features**—the "debiasing" only exists in how the classifier combines these features, not in the features themselves.

This explains why multi-layer distillation makes things worse: earlier layers are even more biased, encoding low-level spurious correlations like background texture.

### Why Do Class Labels Hurt?

With high-quality teachers (91-94% WGA), adding ground-truth labels (α < 1) consistently hurts performance. This is counterintuitive—shouldn't more supervision help?

The issue is that cross-entropy loss treats all samples equally, while AGRE-KD's gradient weighting specifically upweights minority group samples. Adding uniform label supervision dilutes this adaptive weighting, reducing the emphasis on hard minority samples that AGRE-KD is designed to handle.

### Why Doesn't Disagree-Weighting Help?

All five teachers share the same ImageNet-pretrained ResNet-50 backbone. DFR only retrains the final layer differently for each teacher, so their penultimate features are nearly identical. There's simply not enough feature disagreement to exploit.

---

## Conclusion

### Summary of Findings

| Experiment                    | Finding                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------- |
| Feature distillation (γ > 0) | Modest improvement (+0.95% WGA) with reduced variance                            |
| Disagreement weighting        | No benefit—teachers share the same backbone, so feature disagreement is minimal |
| Multi-layer distillation      | Hurts performance—earlier layers encode spurious low-level features             |
| Class labels (α < 1)         | Hurts performance—confirms AGRE-KD appendix finding that pure KD is best        |

### Practical Recommendations

For practitioners using AGRE-KD with DFR-debiased teachers:

- **α = 1.0**: Pure KD, no class labels
- **γ = 0.5**: Optimal feature weight (also lowest variance)
- **Single layer**: Distill from penultimate layer only

### Contribution

This work provides empirical evidence for why the AGRE-KD authors left feature distillation for future work: **the gains are modest when teachers are DFR-debiased**. The fundamental limitation is that DFR only debiases the classifier, not the features being distilled. This constitutes a valid "negative result" that clarifies the boundaries of feature distillation for group-robust KD.

### Limitations

- Used 5 teachers vs. paper's 10 (may affect results)
- Single dataset (Waterbirds)
- Our baseline WGA (84.62%) is lower than the paper's reported ~87.9%

### Future Work

Feature distillation might help more when:

1. Teachers are trained with methods that debias the full backbone (not just classifier)
2. Teachers have diverse architectures with genuinely different features
3. Teachers are biased differently, creating meaningful feature disagreement

---

## References

[1] Kenfack, P.J., et al. "AGRE-KD: Adaptive Group Robust Ensemble Knowledge Distillation." TMLR, 2025.

[2] Kirichenko, P., et al. "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations." NeurIPS, 2022.

[3] Hinton, G., et al. "Distilling the Knowledge in a Neural Network." NIPS Workshop, 2015.

[4] Sagawa, S., et al. "Distributionally Robust Neural Networks for Group Shifts." ICLR, 2020.

---

## Appendix: Ablation Studies

### A1: Full γ Sweep (AGRE-KD)

| γ             | WGA (%)                 | Variance      | n |
| -------------- | ----------------------- | ------------- | - |
| 0.00           | 84.62 ± 1.06           | High          | 4 |
| 0.05           | 83.96                   | —            | 1 |
| 0.10           | 85.10                   | —            | 1 |
| 0.25           | 85.20 ± 0.97           | Medium        | 4 |
| **0.50** | **85.57 ± 0.36** | **Low** | 3 |
| 0.75           | 84.89 ± 0.56           | Low           | 3 |
| 1.00           | 85.10 ± 1.03           | High          | 3 |

### A2: AGRE-KD vs AVER Across γ

| γ   | AGRE-KD | AVER   | Δ     |
| ---- | ------- | ------ | ------ |
| 0.00 | 84.62%  | 83.95% | +0.67% |
| 0.25 | 85.20%  | 84.42% | +0.78% |
| 0.50 | 85.57%  | 84.89% | +0.68% |
| 1.00 | 85.10%  | 84.27% | +0.83% |

AGRE-KD advantage is consistent (~0.7%) across all γ values.

### A3: Combined (α < 1) Full Results

| α  | γ   | WGA (%) | Notes                              |
| --- | ---- | ------- | ---------------------------------- |
| 1.0 | 0.00 | 84.62%  | Baseline                           |
| 0.9 | 0.00 | 83.80%  | Labels hurt                        |
| 0.9 | 0.25 | 84.16%  | Combined worse than features alone |
| 0.7 | 0.00 | 82.55%  | More labels = worse                |
| 0.7 | 0.10 | 83.18%  | Worst combined config              |

### A4: Seed Variance

| Config   | Seed 42 | Seed 43 | Seed 44 | Std     |
| -------- | ------- | ------- | ------- | ------- |
| Baseline | 84.58%  | 85.67%  | 83.18%  | ±1.06% |
| γ=0.50  | 85.36%  | 85.98%  | 85.36%  | ±0.36% |

γ=0.5 produces more consistent results across random seeds.
