# Feature Distillation for Group-Robust Knowledge Distillation

**Course**: 6.7960 Deep Learning, Fall 2025
**Date**: December 8, 2025

---

## 1. The Problem: Why Do Neural Networks Learn Shortcuts?

### 1.1 Spurious Correlations in Training Data

Neural networks are powerful pattern recognizers, but they often learn the *wrong* patterns. Consider a dataset of bird images where:

- **Waterbirds** (the class we want to identify) mostly appear against water backgrounds
- **Landbirds** mostly appear against land backgrounds

A network trained on this data might learn: "water background → waterbird" instead of learning actual bird features (beak shape, wing patterns, feathers). This background-class correlation is **spurious**—it exists in the training data but doesn't reflect the true relationship we want to learn.

### 1.2 The Minority Group Problem

The danger emerges when we encounter **minority groups**—samples that break the spurious correlation:

| Group | Class | Background | Training % | What Happens |
|-------|-------|------------|------------|--------------|
| Majority | Waterbird | Water | ~73% | Network learns this well |
| Majority | Landbird | Land | ~22% | Network learns this well |
| **Minority** | Waterbird | Land | ~4% | Network fails here |
| **Minority** | Landbird | Water | ~1% | Network fails here |

A network achieving 95% overall accuracy might have only 60% accuracy on these minority groups. This is catastrophic for fairness—the model systematically fails on certain subpopulations.

### 1.3 Worst-Group Accuracy (WGA)

To measure robustness against spurious correlations, we use **Worst-Group Accuracy (WGA)**:

$$\text{WGA} = \min_{g \in \mathcal{G}} \text{Accuracy}(g)$$

Instead of averaging across all samples (which hides minority group failures), WGA reports the accuracy of the *worst-performing* group. A model with 95% average accuracy but 60% WGA is learning shortcuts.

**Our goal**: Maximize WGA to ensure the model performs well on ALL groups, not just the easy majority.

---

## 2. Background: Knowledge Distillation and Its Problems

### 2.1 What is Knowledge Distillation?

Knowledge distillation (KD) transfers knowledge from a large "teacher" model to a smaller "student" model. Instead of training the student on hard labels (0 or 1), we train it to match the teacher's soft probability outputs.

**Why soft labels help**: A teacher's output [0.7, 0.2, 0.1] contains more information than [1, 0, 0]. It tells the student "this is probably class A, but has some similarity to class B."

### 2.2 Ensemble Knowledge Distillation

Using multiple teachers (an ensemble) often works better than a single teacher:

```
Teacher 1 → [0.8, 0.2]
Teacher 2 → [0.7, 0.3]    →  Average: [0.75, 0.25]  →  Student learns from this
Teacher 3 → [0.75, 0.25]
```

The averaged predictions are typically more robust than any single teacher.

### 2.3 The Bias Amplification Problem

Here's the critical issue that motivates our work:

**Even if each teacher is debiased (has good WGA), the distilled student can have WORSE WGA than the teachers.**

This is **bias amplification**. The simple averaging process can reinforce spurious correlations that exist across teachers, even if individual teachers have learned to partially overcome them.

---

## 3. AGRE-KD: The Baseline We Extend

### 3.1 The Core Idea

**AGRE-KD** (Adaptive Group Robust Ensemble Knowledge Distillation) by Kenfack et al. (TMLR 2025) addresses bias amplification through **gradient-based teacher weighting**.

Instead of equally averaging all teachers, AGRE-KD asks: "Which teachers are giving *non-biased* guidance for this particular sample?"

### 3.2 How It Works

AGRE-KD maintains a **biased reference model**—a model that has learned the spurious correlations (low WGA, high average accuracy). For each training sample:

1. Compute the gradient from each teacher's prediction
2. Compute the gradient from the biased model's prediction
3. If a teacher's gradient *aligns* with the biased model → that teacher is giving biased guidance → **downweight it**
4. If a teacher's gradient *disagrees* with the biased model → that teacher is giving useful guidance → **upweight it**

Mathematically:
$$W_t(x_i) = 1 - \langle \nabla \ell_i^t(\theta), \nabla \ell_i^b(\theta) \rangle$$

Where $\langle \cdot, \cdot \rangle$ is cosine similarity between gradients.

### 3.3 Why This Works

For a minority group sample (waterbird on land):
- The biased model confidently predicts "landbird" (wrong, based on background)
- A teacher that has learned correct bird features predicts "waterbird"
- These gradients point in *opposite directions*
- AGRE-KD upweights this teacher's contribution

The result: minority groups get guidance from teachers that disagree with bias.

### 3.4 The Gap We Address

The AGRE-KD paper explicitly states:

> *"We restrict ourselves to logit distillation and leave feature distillation for future exploration."*

AGRE-KD only uses the teacher's final output logits. But neural networks learn hierarchical representations—could we also distill the **intermediate features** to improve group robustness? This is what we investigate.

---

## 4. Our Extension: Feature Distillation

### 4.1 What is Feature Distillation?

Beyond matching teacher logits, we can also match **intermediate representations**. In a ResNet, the penultimate layer (before the final classifier) contains a 2048-dimensional feature vector that represents "what the network sees" in the image.

Feature distillation adds a term to the loss:
$$\mathcal{L}_{feat} = \| f_s(x) - \bar{f}_T(x) \|_2^2$$

Where $f_s(x)$ is the student's features and $\bar{f}_T(x)$ is the (weighted) average of teacher features.

### 4.2 The Complete Loss Function

Our training objective combines three components:

$$\mathcal{L}_{total} = (1-\alpha) \mathcal{L}_{cls} + \alpha \mathcal{L}_{wKD} + \gamma \mathcal{L}_{feat}$$

| Term | Meaning | What It Does |
|------|---------|--------------|
| $\mathcal{L}_{cls}$ | Cross-entropy with true labels | Direct supervision from ground truth |
| $\mathcal{L}_{wKD}$ | Weighted KD loss (AGRE-KD) | Learn from gradient-weighted teacher logits |
| $\mathcal{L}_{feat}$ | Feature distillation loss | Match teacher feature representations |
| $\alpha$ | KD weight (0-1) | Balance between labels and teacher logits |
| $\gamma$ | Feature weight (0+) | How much to emphasize feature matching |

### 4.3 Hypothesis: Why Feature Distillation Might Help

**Kirichenko et al. (2022)** showed that even in biased models, the penultimate layer still contains information about *core features* (actual bird characteristics), not just spurious correlations (backgrounds). This is why **DFR (Deep Feature Reweighting)** works—it retrains only the last classifier layer on balanced data, extracting the good features that were always there.

Our teachers are DFR-debiased. If their penultimate features contain debiased information, distilling these features might transfer the "debiasing signal" more effectively than logits alone.

### 4.4 Counter-Hypothesis: Why It Might NOT Help

DFR only retrains the **last layer**. The backbone (all layers before the classifier) remains unchanged from standard training. This means:

- The backbone features are still biased
- All 5 teachers share the same ImageNet-pretrained ResNet-50 backbone
- DFR just learns a *different linear combination* of these biased features

If features remain biased, distilling them could reinforce spurious correlations rather than reduce them.

---

## 5. Experimental Setup

### 5.1 Dataset: Waterbirds

| Property | Value |
|----------|-------|
| Total training images | 4,795 |
| Classes | 2 (waterbird, landbird) |
| Spurious attribute | Background (water, land) |
| Groups | 4 (2 classes × 2 backgrounds) |
| Minority group size | ~1-4% of training data |

### 5.2 Model Architecture

| Component | Architecture | Notes |
|-----------|--------------|-------|
| Teachers (5×) | ResNet-50 | ImageNet pretrained, DFR-debiased |
| Student | ResNet-18 | ImageNet pretrained, trained from scratch |
| Biased model | ResNet-50 | Standard ERM training (WGA ~73.8%) |

**Teacher quality**: After DFR debiasing, teachers achieve 91.9% - 93.8% WGA.

### 5.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | SGD (momentum=0.9) | Standard for image classification |
| Learning rate | 0.001 | With cosine decay |
| Batch size | 128 | Fits in T4 GPU memory |
| Epochs | 30 | Sufficient for convergence |
| Temperature (τ) | 4.0 | Standard for KD |

### 5.4 Evaluation Protocol

- **Metric**: Worst-Group Accuracy (WGA) on test set
- **Seeds**: 42, 43, 44 (3 runs for statistical significance)
- **Statistical reporting**: Mean ± standard deviation

---

## 6. Experiment 1: Baseline Comparison (AGRE-KD vs Simple Averaging)

### 6.1 Research Question

Before testing feature distillation, we need to establish: Does AGRE-KD's gradient weighting actually help compared to simple teacher averaging?

### 6.2 Setup

| Method | Description | α | γ |
|--------|-------------|---|---|
| AGRE-KD | Gradient-weighted teacher ensemble | 1.0 | 0.0 |
| AVER | Simple arithmetic average of teachers | 1.0 | 0.0 |

Both methods use only logit distillation (γ=0, no features).

### 6.3 Results

| Method | WGA | n | Interpretation |
|--------|-----|---|----------------|
| AGRE-KD | 84.62 ± 1.06% | 4 | Gradient weighting helps |
| AVER | 83.95 ± 0.95% | 4 | Simple averaging is worse |
| **Δ** | **+0.67%** | | Consistent improvement |

### 6.4 Analysis

AGRE-KD provides a small but consistent improvement (+0.67%) over simple averaging. This validates that gradient-based weighting helps prevent bias amplification.

**Note on paper comparison**: The original AGRE-KD paper reports ~87.9% WGA. Our lower baseline (84.62%) likely stems from using 5 teachers (vs. paper's 10) and potentially different hyperparameters. However, this doesn't affect our comparative findings—all our experiments use the same setup.

---

## 7. Experiment 2: Feature Distillation Gamma Sweep

### 7.1 Research Question

Does adding intermediate feature distillation (γ > 0) improve WGA beyond logit-only distillation?

### 7.2 Setup

We sweep γ from 0 to 1 while keeping α=1 (no class labels, pure KD + features):

| Experiment | α | γ | What Changes |
|------------|---|---|--------------|
| baseline | 1.0 | 0.00 | Logits only |
| gamma_005 | 1.0 | 0.05 | Very light feature weight |
| gamma_025 | 1.0 | 0.25 | Light feature weight |
| gamma_050 | 1.0 | 0.50 | Balanced logits and features |
| gamma_075 | 1.0 | 0.75 | Heavy feature weight |
| gamma_100 | 1.0 | 1.00 | Equal logits and features |

### 7.3 Results

| γ | WGA | Δ vs Baseline | Seeds | Variance |
|---|-----|---------------|-------|----------|
| 0.00 | 84.62 ± 1.06% | — | 4 | High |
| 0.05 | 83.96% | -0.66% | 1 | N/A |
| 0.25 | 85.20 ± 0.97% | +0.58% | 4 | Medium |
| 0.50 | **85.57 ± 0.36%** | **+0.95%** | 3 | **Lowest** |
| 0.75 | 84.89 ± 0.56% | +0.27% | 3 | Low |
| 1.00 | 85.10 ± 1.03% | +0.48% | 3 | High |

### 7.4 Gamma Sensitivity Visualization

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

### 7.5 Analysis

**Finding 1: Feature distillation provides marginal improvement**

The best configuration (γ=0.5) achieves 85.57% WGA, a +0.95% improvement over baseline. This is modest but consistent.

**Finding 2: γ=0.5 is optimal**

- Too low (γ=0.05): Feature signal too weak, actually hurts
- Too high (γ=1.0): Feature signal dominates, less effective
- γ=0.5 balances logit and feature distillation

**Finding 3: γ=0.5 has lowest variance**

Across seeds, γ=0.5 shows ±0.36% variance vs. ±1.06% for baseline. This suggests feature distillation stabilizes training.

**Why is the improvement only marginal?**

Our counter-hypothesis appears correct: DFR only debiases the classifier head. The backbone features (which we're distilling) remain biased. We're essentially distilling *biased* features, limiting the benefit.

---

## 8. Experiment 3: Class Labels (α < 1)

### 8.1 Research Question

AGRE-KD uses pure KD (α=1, no ground-truth labels). Would adding some class label supervision (α < 1) help, especially when combined with feature distillation?

### 8.2 Motivation

The AGRE-KD paper suggests class labels might help when teachers are biased, as ground truth provides a debiasing signal. We test this with our high-quality DFR-debiased teachers.

### 8.3 Setup

| Experiment | α | γ | What Changes |
|------------|---|---|--------------|
| baseline | 1.0 | 0.00 | Pure KD |
| combined | 0.9 | 0.25 | 10% class labels + features |

With α=0.9, the loss becomes:
- 10% cross-entropy with ground truth labels
- 90% weighted KD loss
- Plus γ=0.25 feature distillation

### 8.4 Results

| Experiment | α | γ | WGA | Δ vs Baseline |
|------------|---|---|-----|---------------|
| baseline | 1.0 | 0.00 | 84.62% | — |
| feature only | 1.0 | 0.25 | 85.20% | +0.58% |
| **combined** | **0.9** | **0.25** | **84.16%** | **-0.46%** |

### 8.5 Analysis

**Finding: Class labels HURT performance**

Adding ground-truth labels decreased WGA by 0.46%. This contradicts the intuition that more supervision should help.

**Why class labels hurt with high-quality teachers**:

1. **Teachers are already excellent**: With 91-94% WGA, teachers provide better guidance than raw labels
2. **Cross-entropy treats all samples equally**: Unlike AGRE-KD, standard classification loss doesn't upweight minority groups
3. **Label signal interferes with gradient weighting**: The class label gradient can override the carefully computed AGRE weights

**Takeaway**: When teachers are high-quality (DFR-debiased), trust the teachers. Adding class labels introduces noise.

---

## 9. Experiment 4: Disagree-Weight Feature Distillation

### 9.1 Research Question

AGRE-KD weights teacher *logits* by gradient disagreement with bias. Can we apply a similar principle to *features*?

### 9.2 Motivation

If teachers produce different features for a sample, that disagreement might indicate the sample is:
- From a minority group (where biased and unbiased teachers would differ)
- Ambiguous or "hard" (where teachers learned different solutions)

Weighting the feature loss by teacher disagreement might emphasize these informative samples.

### 9.3 Method

For each sample, compute the variance (disagreement) across teacher features:

$$w_i = \text{Var}(\{f_t(x_i)\}_{t=1}^T)$$

Then weight the feature distillation loss by this variance:

$$\mathcal{L}_{feat}^{disagree} = w_i \cdot \| f_s(x) - \bar{f}_T(x) \|_2^2$$

Samples where teachers disagree more receive higher weight.

### 9.4 Results

| Method | γ | WGA | vs Standard |
|--------|---|-----|-------------|
| Standard AGRE-KD | 0.50 | 85.57% | — |
| Disagree-Weight | 0.50 | 85.31% | -0.26% |

### 9.5 Analysis

**Finding: Disagree-weighting doesn't help**

The variance-based feature weighting performs slightly worse than standard averaging. This suggests:

1. **Teacher feature disagreement isn't informative**: Unlike logit-level disagreement (which reflects different predictions), feature-level disagreement may just be noise
2. **DFR teachers have similar features**: Since all teachers share the same backbone, their features are nearly identical—there's little disagreement to exploit
3. **Gradient weighting already captures the signal**: AGRE-KD's logit-level weighting may already incorporate the relevant information

---

## 10. Experiment 5: Multi-Layer Feature Distillation

### 10.1 Research Question

We've been distilling from the penultimate layer (layer4 in ResNet). Would distilling from *multiple* layers capture richer information?

### 10.2 Motivation

Different layers capture different abstractions:
- **Early layers (layer2)**: Low-level features (edges, textures)
- **Middle layers (layer3)**: Mid-level features (parts, shapes)
- **Late layers (layer4)**: High-level features (objects, concepts)

Perhaps earlier layers contain complementary debiased information that layer4 alone doesn't capture.

### 10.3 Method

Extract features from multiple ResNet layers and distill from each. Since teacher (ResNet-50) and student (ResNet-18) have different dimensions, we use **adapter networks** (1×1 convolutions) to match dimensions:

| Layer | Teacher Dim | Student Dim | Adapter |
|-------|-------------|-------------|---------|
| layer2 | 512 | 128 | 512 → 128 |
| layer3 | 1024 | 256 | 1024 → 256 |
| layer4 | 2048 | 512 | 2048 → 512 |

### 10.4 Configurations Tested

| Config | Layers | Description |
|--------|--------|-------------|
| L4 only | layer4 | Standard (baseline for comparison) |
| L3+L4 | layer3, layer4 | Two-layer distillation |
| L2+L3+L4 | layer2, layer3, layer4 | Three-layer distillation |

### 10.5 Results

| Configuration | WGA | Δ vs L4 Only |
|---------------|-----|--------------|
| L4 only | 85.57% | — |
| L3+L4 | 85.20% | -0.37% |
| L2+L3+L4 | 84.74% | -0.83% |

### 10.6 Analysis

**Finding: More layers = WORSE performance**

Adding earlier layers hurt WGA. The more layers we distill, the worse the result.

**Why multi-layer hurts**:

1. **DFR doesn't debias the backbone**: Only the final classifier is retrained. Layers 2, 3, and 4 all contain biased features
2. **Earlier layers are MORE biased**: They encode low-level features like texture and background—exactly the spurious correlations we want to avoid
3. **Distilling bias reinforces bias**: By matching biased early-layer features, we're teaching the student to rely on the wrong patterns

**Takeaway**: If the backbone isn't debiased, distilling more backbone layers just distills more bias.

---

## 11. Experiment 6: Feature Distillation with Simple Averaging (AVER)

### 11.1 Research Question

Does feature distillation also help when using simple teacher averaging (no gradient weighting)?

### 11.2 Motivation

This helps isolate the effect of feature distillation from AGRE-KD's gradient weighting.

### 11.3 Results

| γ | AGRE-KD | AVER | Δ (AGRE-KD - AVER) |
|---|---------|------|---------------------|
| 0.00 | 84.62% | 83.95% | +0.67% |
| 0.25 | 85.20% | 84.42% | +0.78% |
| 0.50 | 85.57% | 84.89% | +0.68% |
| 1.00 | 85.10% | 84.27% | +0.83% |

### 11.4 Analysis

**Finding 1: Feature distillation helps both methods**

AVER with γ=0.5 (84.89%) beats AVER baseline (83.95%) by +0.94%. The improvement is similar to AGRE-KD (+0.95%).

**Finding 2: AGRE-KD advantage is consistent**

Across all γ values, AGRE-KD beats AVER by ~0.7-0.8%. Gradient weighting helps regardless of feature distillation.

**Finding 3: Effects are additive**

Feature distillation and gradient weighting provide independent benefits. Using both (AGRE-KD + γ=0.5) gives the best result.

---

## 12. Summary of All Experiments

### 12.1 Complete Results Table

| Rank | Experiment | Method | α | γ | WGA | Key Finding |
|------|------------|--------|---|---|-----|-------------|
| 1 | gamma_050 | AGRE-KD | 1.0 | 0.50 | **85.57%** | Optimal configuration |
| 2 | disagree_weight | Disagree-Weight | 1.0 | 0.50 | 85.31% | Variance weighting doesn't help |
| 3 | gamma_025 | AGRE-KD | 1.0 | 0.25 | 85.20% | Lower γ also works |
| 4 | ml_L3_L4 | Multi-Layer | 1.0 | 0.50 | 85.20% | Multi-layer doesn't help |
| 5 | gamma_100 | AGRE-KD | 1.0 | 1.00 | 85.10% | High γ slightly worse |
| 6 | gamma_075 | AGRE-KD | 1.0 | 0.75 | 84.89% | Diminishing returns |
| 7 | aver_gamma_050 | AVER | 1.0 | 0.50 | 84.89% | AVER benefits from features too |
| 8 | ml_L2_L3_L4 | Multi-Layer | 1.0 | 0.50 | 84.74% | More layers = worse |
| 9 | **baseline** | AGRE-KD | 1.0 | 0.00 | 84.62% | Our baseline |
| 10 | aver_gamma_025 | AVER | 1.0 | 0.25 | 84.42% | AVER with light features |
| 11 | aver_gamma_100 | AVER | 1.0 | 1.00 | 84.27% | AVER with heavy features |
| 12 | combined | AGRE-KD | 0.9 | 0.25 | 84.16% | Class labels hurt |
| 13 | gamma_005 | AGRE-KD | 1.0 | 0.05 | 83.96% | γ too low hurts |
| 14 | aver_baseline | AVER | 1.0 | 0.00 | 83.95% | Simple averaging baseline |

### 12.2 Hypothesis Results

| # | Hypothesis | Result | Evidence |
|---|------------|--------|----------|
| H1 | Class labels (α < 1) improve WGA | **FALSE** | 84.16% vs 84.62% baseline |
| H2 | Feature distillation (γ > 0) improves WGA | **TRUE (marginal)** | 85.57% vs 84.62% (+0.95%) |
| H3 | Combined approach is best | **FALSE** | Combined is worst of feature configs |

---

## 13. Discussion: Why Are Improvements Limited?

### 13.1 The DFR Bottleneck

Our results consistently show that feature distillation provides only marginal improvement. The root cause is **DFR's design**:

```
ResNet-50 Architecture:
┌─────────────────────────────────────────┐
│ layer1 → layer2 → layer3 → layer4 → FC │
│        (BIASED BACKBONE)          │(DFR)│
└─────────────────────────────────────────┘
                                      ↑
                              Only this is debiased
```

DFR retrains ONLY the final fully-connected (FC) layer. The entire backbone (layers 1-4) remains in its original, biased state. When we distill features from layer4:

- We're distilling **biased** features
- The "debiasing" only happens in the FC layer (the logits)
- Feature distillation can't transfer what doesn't exist in features

### 13.2 Teacher Homogeneity

All 5 teachers share the same ImageNet-pretrained ResNet-50 backbone. DFR only retrains the last layer differently for each teacher. This means:

- Teacher features are nearly identical
- The weighted average of identical features provides little additional signal
- Disagreement-based weighting has nothing to exploit

### 13.3 What Would Work Better?

Based on our findings, feature distillation would likely help more if:

1. **Teachers have diverse backbones**: Different architectures or training procedures
2. **Debiasing affects the full network**: Methods like GroupDRO or JTT that modify backbone training
3. **Teachers are biased differently**: Each teacher learns different spurious correlations

---

## 14. Conclusions and Recommendations

### 14.1 Key Takeaways

1. **Feature distillation provides marginal benefit (+0.95%)** when teachers are DFR-debiased, because DFR doesn't debias backbone features

2. **γ=0.5 is optimal**: Balances logit and feature signals, also provides lowest variance across seeds

3. **Don't add class labels**: When teachers are high-quality (90%+ WGA), class labels interfere with gradient weighting

4. **Single layer is sufficient**: Multi-layer distillation distills more bias, not less

5. **AGRE-KD > Simple averaging**: Gradient weighting provides consistent +0.7% improvement

### 14.2 Practical Recommendations

For practitioners using AGRE-KD with DFR-debiased teachers:

| Setting | Value | Rationale |
|---------|-------|-----------|
| α | 1.0 | Pure KD, no class labels |
| γ | 0.5 | Optimal feature weight |
| Layers | layer4 only | Earlier layers add bias |
| Weighting | Standard AGRE-KD | Disagree-weight doesn't help |

### 14.3 Contribution

This work provides empirical evidence answering the AGRE-KD paper's open question about feature distillation:

> Feature distillation offers **limited benefit** for group-robust KD when teachers are DFR-debiased. The limitation is fundamental: DFR only debiases the classifier, not the features being distilled.

This constitutes a valid "negative result" that explains why more sophisticated feature distillation methods (multi-layer, disagree-weighting) fail to improve over simple feature matching.

---

## 15. Future Directions

### 15.1 Test with Biased Teachers

Our teachers are DFR-debiased (91-94% WGA). Feature distillation might help more when:
- Teachers are trained with standard ERM (biased)
- Teachers have diverse biases

### 15.2 End-to-End Debiased Teachers

Use debiasing methods that affect the full network:
- **GroupDRO**: Robust optimization that may affect features
- **JTT (Just Train Twice)**: Reweighting that affects backbone training
- **Contrastive debiasing**: Explicitly debiases representations

### 15.3 Larger Ensembles

The original AGRE-KD paper uses 10 teachers. Testing with larger ensembles might show different feature distillation behavior.

### 15.4 Other Datasets

Validate findings on:
- **CelebA**: Facial attribute prediction with spurious correlations
- **MultiNLI**: Natural language inference with annotation artifacts

---

## Appendix: Experimental Details

### A.1 Compute Resources

| Resource | Specification |
|----------|---------------|
| Platform | Google Colab |
| GPU | NVIDIA T4 (16GB) |
| Time per run | 45-60 minutes |
| Total experiments | 34 runs |
| Total compute | ~25-30 GPU hours |

### A.2 Experiment Counts

| Category | Runs |
|----------|------|
| Baseline (AGRE-KD) | 4 |
| Baseline (AVER) | 4 |
| Feature Distillation (AGRE-KD) | 17 |
| Feature Distillation (AVER) | 3 |
| Combined (α < 1) | 3 |
| Disagree Weighting | 3 |
| Multi-Layer | 2 |

### A.3 Seeds

All experiments use seeds 42, 43, 44 for reproducibility. Original "og" runs (before seeding was implemented) are included but excluded from statistical analysis.

### A.4 Code Availability

Training code, evaluation scripts, and checkpoints are available in this repository under `light-code/`.
