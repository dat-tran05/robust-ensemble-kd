# Improving Group Robustness in Ensemble Knowledge Distillation: Beyond Logit Matching

**Course**: 6.7960 Deep Learning, Fall 2025  
**Team Size**: 2 members  
**Timeline**: 5 days  

---

## 1. Executive Summary

This project investigates whether ensemble knowledge distillation can be made more robust to spurious correlations by incorporating (1) ground-truth class labels and (2) intermediate feature distillation into the training objective. We extend AGRE-KD (Adaptive Group Robust Ensemble Knowledge Distillation), a recent method that uses gradient-based teacher weighting to improve worst-group accuracy (WGA), by relaxing its "unsupervised" constraint and adding feature-level supervision.

**Core Research Question**: Can we improve the group robustness of distilled student models by combining logit-based knowledge distillation with class labels and/or intermediate feature matching?

---

## 2. Background and Motivation

### 2.1 The Problem: Spurious Correlations in Deep Learning

Neural networks trained with standard Empirical Risk Minimization (ERM) often learn **spurious correlations**—features that are statistically associated with labels in training data but have no causal relationship to the task. This leads to:

- High average accuracy on test data with similar spurious patterns
- Poor performance on **minority groups** where the spurious correlation doesn't hold

**Example (Waterbirds dataset)**:
- Task: Classify birds as "waterbird" or "landbird"
- Spurious correlation: 95% of waterbirds appear on water backgrounds
- Failure case: Waterbirds on land backgrounds are misclassified because the model learned "water background → waterbird"

| Group | Description | Training % | Model Behavior |
|-------|-------------|------------|----------------|
| Waterbird + Water | Majority | ~73% | High accuracy |
| Landbird + Land | Majority | ~22% | High accuracy |
| Waterbird + Land | **Minority** | ~1% | **Low accuracy** |
| Landbird + Water | Minority | ~4% | Low accuracy |

The standard metric for evaluating robustness to spurious correlations is **Worst-Group Accuracy (WGA)**—the minimum accuracy across all subgroups.

### 2.2 Knowledge Distillation: Compressing Models

Knowledge Distillation (KD) transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to match the teacher's soft probability outputs (logits) rather than just hard labels:

$$\mathcal{L}_{KD} = \text{KL}\left( \sigma\left(\frac{z_s}{\tau}\right) \| \sigma\left(\frac{z_t}{\tau}\right) \right)$$

Where:
- $z_s, z_t$ are student and teacher logits
- $\tau$ is temperature (higher = softer probabilities)
- $\sigma$ is softmax function

The "dark knowledge" in soft labels captures inter-class relationships (e.g., a cat is more similar to a dog than to a truck) that hard labels don't provide.

### 2.3 The Problem: Bias Amplification in Knowledge Distillation

Recent research has revealed a concerning phenomenon: **knowledge distillation can amplify bias**. Even when teacher models are debiased, the distilled student often performs worse on minority groups than the teachers.

This occurs because:
1. Students have lower capacity than teachers, forcing them to learn simpler (often spurious) features
2. Soft labels from biased teachers encode the spurious correlations
3. The distillation objective optimizes for average performance, not worst-group performance

### 2.4 Ensemble Methods: A Partial Solution

**Deep Ensembles** (multiple models trained with different random seeds) naturally improve group robustness because:
- Different models may learn different features
- Averaging predictions reduces reliance on any single spurious pattern
- Diversity in the ensemble captures more robust representations

However, ensembles are expensive at inference time. **Ensemble Knowledge Distillation** aims to distill the collective knowledge of multiple teachers into a single student—but traditional methods (simple averaging) still amplify bias.

---

## 3. The AGRE-KD Baseline

### 3.1 Method Overview

**AGRE-KD** (Adaptive Group Robust Ensemble Knowledge Distillation) addresses bias amplification in ensemble KD through gradient-based teacher weighting:

1. Train an ensemble of teachers with standard ERM (different random seeds)
2. Optionally debias teachers using DFR (Deep Feature Reweighting)
3. Select one ERM teacher as a "biased reference model"
4. During student training, weight each teacher's contribution based on how much their gradient **disagrees** with the biased model

**Key Insight**: Teachers whose gradients point in the same direction as the biased model are likely reinforcing spurious correlations. By downweighting them, the student learns more robust features.

### 3.2 The Weighting Scheme

For each sample $x_i$ and teacher $t$, compute:

$$W_t(x_i) = 1 - \langle \nabla \ell_i^t(\theta), \nabla \ell_i^b(\theta) \rangle$$

Where:
- $\nabla \ell_i^t(\theta)$ is the normalized gradient of KD loss w.r.t. teacher $t$
- $\nabla \ell_i^b(\theta)$ is the normalized gradient of KD loss w.r.t. biased model
- The dot product measures gradient alignment

Teachers aligned with the biased model get low weights; teachers pointing away from bias get high weights.

### 3.3 The Loss Function

$$\mathcal{L} = \alpha \mathcal{L}_{wKD} + (1-\alpha) \mathcal{L}_{cls}$$

Where:
- $\mathcal{L}_{wKD}$ is the weighted KD loss (aggregated across teachers)
- $\mathcal{L}_{cls}$ is the standard cross-entropy loss with ground-truth labels

**Critical Design Choice**: The paper sets **α = 1**, meaning they use **only** teacher outputs and **no class labels**. They call this "unsupervised knowledge distillation."

### 3.4 Limitations Identified by the Authors

From the AGRE-KD conclusion:

> "First, this study focused on unsupervised KD using teachers' logits alone, without access to group or class labels. Second, WGA improvements are less pronounced when all teachers in the ensemble are biased, suggesting an opportunity to **exploit class labels** in this setting to further boost WGA."

Additionally, the paper only uses **logit distillation**—it does not explore whether matching intermediate layer representations could improve robustness.

---

## 4. Our Proposed Extensions

We propose three experiments that address the identified limitations:

### 4.1 Experiment 1: Incorporating Class Labels (α < 1)

**Hypothesis**: Adding ground-truth supervision alongside teacher distillation will improve WGA, especially when teachers are biased.

**Rationale**:
- Class labels provide a "ground truth anchor" that prevents the student from fully adopting teacher biases
- When teachers make mistakes on minority groups, the classification loss can correct this
- This is a direct test of the paper's stated limitation

**Implementation**:
- Use the same AGRE-KD weighting scheme
- Set α ∈ {0.5, 0.7, 0.9} instead of α = 1
- Compare WGA across different α values

### 4.2 Experiment 2: Feature Distillation (No Class Labels)

**Hypothesis**: Distilling penultimate layer representations, rather than just logits, will transfer more robust features to the student by aligning representations at the layer where core features are maximally decodable.

**Rationale**:

| Reason | Explanation |
|--------|-------------|
| **Core features are decodable at penultimate layer** | Kirichenko et al. (ICLR 2023) showed that even when models achieve 0% worst-group accuracy, core features can still be decoded from penultimate representations using simple logistic regression |
| **Alignment with DFR** | Our teachers are debiased via DFR, which retrains the classifier on penultimate features. The "debiasing signal" lives in how the classifier interprets these features—distilling at this layer captures the representation space where debiasing operates |
| **Avoids spurious-heavy early layers** | Research shows early/shallow layers predominantly encode spurious features (backgrounds, textures). Distilling from layer1 or layer2 could actively transfer spurious representations |
| **Semantic richness** | Penultimate features represent the final abstraction before classification—full object representations rather than edges or textures |

**Why One Layer is Sufficient**:

Research from Yu et al. (2025) tested multiple layer-matching strategies and found that **all strategies performed surprisingly similarly**—the presence of intermediate matching matters more than which specific layers you choose. Additionally:

- The first intermediate layer provides most of the benefit (~1.5-2.0% gain)
- Each additional layer adds diminishing returns (+0.3-0.5% at most)
- Since our DFR teachers have identical features (only the classifier differs), feature aggregation across teachers essentially averages nearly-identical representations

**Implementation**:
- Keep α = 1 (no class labels, to isolate the effect of feature distillation)
- Add a feature matching loss at the **penultimate layer (layer4)** of ResNet:

$$\mathcal{L}_{feat} = \| \text{Adapter}(f_s^{(4)}) - f_t^{(4)} \|^2$$

- Final loss: $\mathcal{L} = \mathcal{L}_{wKD} + \gamma \mathcal{L}_{feat}$ with γ ∈ {0.1, 0.25}
- **Extension if time permits**: Add layer3 as secondary distillation point to test diminishing returns

### 4.3 Experiment 3: Combined Approach

**Hypothesis**: Combining class labels AND feature distillation will achieve the best WGA by leveraging both ground-truth supervision and representation-level knowledge transfer.

**Implementation**:

$$\mathcal{L}_{total} = (1-\alpha) \mathcal{L}_{cls} + \alpha \mathcal{L}_{wKD} + \gamma \mathcal{L}_{feat}$$

With α ∈ {0.7, 0.9} and γ ∈ {0.1, 0.25}

### 4.4 Summary of Experimental Conditions

| Experiment | Class Labels (α<1) | Feature Distillation (γ>0) | What We're Testing |
|------------|-------------------|---------------------------|-------------------|
| Baseline (AGRE-KD) | ✗ (α=1) | ✗ (γ=0) | Replication |
| Exp 1 | ✓ | ✗ | Effect of class labels |
| Exp 2 | ✗ | ✓ | Effect of feature distillation |
| Exp 3 | ✓ | ✓ | Combined effect |

---

## 5. Experimental Design

### 5.1 Datasets

**Waterbirds** (Primary)
- 4,795 training images, 1,199 test images
- 2 classes × 2 backgrounds = 4 groups
- Hardest group: waterbird on land (56 training samples, 1.2%)
- Fast to train (~1-2 min/epoch on T4)

**CelebA Blonde Hair** (Secondary, if time permits)
- 162,770 training images
- Task: Predict blonde hair
- Spurious correlation: blonde hair correlates with female gender
- Hardest group: blonde males (~1% of training data)
- Slower to train (~8-15 min/epoch on T4)

### 5.2 Model Architecture

- **Teachers**: ResNet-50 pretrained on ImageNet (3-5 models with different seeds)
- **Student**: ResNet-18 pretrained on ImageNet
- **Feature distillation layer**: **Penultimate layer (layer4)** output
  - ResNet-18 layer4: 512 channels
  - ResNet-50 layer4: 2048 channels
  - Requires a 1×1 conv adapter to align dimensions (512 → 2048)
  
**Why penultimate layer?** This is where DFR operates and where core features are maximally decodable (Kirichenko et al., 2023). Early layers encode spurious features; the penultimate layer encodes full object representations.

### 5.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.001 (with cosine decay) |
| Weight decay | 1e-4 |
| Batch size | 32 (Waterbirds), 64 (CelebA) |
| Epochs | 50 (teachers), 30 (student) |
| Temperature (τ) | 4 |

### 5.4 Evaluation Metrics

**Primary Metrics**:
1. **Worst-Group Accuracy (WGA)**: min_{g} Accuracy(g) — the key metric for group robustness
2. **Average Accuracy**: Overall test accuracy

**Secondary Metrics**:
3. **Per-group accuracy breakdown**: Accuracy for each of the 4 groups
4. **Accuracy Gap**: max_{g} Accuracy(g) - min_{g} Accuracy(g)

### 5.5 Baselines for Comparison

| Method | Description | Expected WGA (Waterbirds) |
|--------|-------------|--------------------------|
| ERM | Standard training, no debiasing | 68-72% |
| Deep Ensemble | Average predictions of 3-5 models | 75-80% |
| AVER | Simple average of teacher logits | ~70% |
| AE-KD | Adaptive Ensemble KD (gradient-based) | ~75% |
| AGRE-KD | Our baseline (logit-only, α=1) | ~85-88% |
| **Ours** | AGRE-KD + class labels + features | ? |

---

## 6. Expected Outcomes and Contributions

### 6.1 Possible Results

**Positive outcome**: Our extensions improve WGA over AGRE-KD
- Class labels help anchor learning when teachers are wrong
- Feature distillation transfers more robust intermediate representations
- Combined approach achieves state-of-the-art WGA

**Negative outcome**: Our extensions don't help or hurt WGA
- This would still be valuable: we'd explain *why* these intuitive approaches fail
- Possible reasons: feature distillation also transfers spurious features; class labels conflict with teacher guidance

**Mixed outcome**: One extension helps, the other doesn't
- Most likely scenario
- Provides nuanced insights about when each approach is beneficial

### 6.2 Scientific Contributions

Regardless of outcome, this project contributes:

1. **Empirical analysis** of class label incorporation in ensemble KD for debiasing
2. **First exploration** of feature distillation for group-robust ensemble KD
3. **Ablation study** isolating the effects of each component
4. **Practical recommendations** for practitioners choosing KD strategies

### 6.3 Testable Hypotheses (for Blog Post)

1. **H1**: Setting α < 1 (adding class labels) improves WGA over α = 1
2. **H2**: Adding penultimate-layer feature distillation (γ > 0) improves WGA over γ = 0
3. **H3**: The combined approach (α < 1, γ > 0) achieves higher WGA than either alone
4. **H4**: Feature distillation helps more when teachers are biased (not debiased with DFR)
5. **H5** (if time permits): Adding layer3 distillation provides marginal additional gains over penultimate-only (testing diminishing returns)

---

## 7. Timeline and Division of Work

### Day-by-Day Plan

| Day | Person A | Person B |
|-----|----------|----------|
| 1 | Setup codebase, data loaders | Train Teacher 1 & 2 (Waterbirds) |
| 2 | Implement AGRE-KD weighting | Train Teacher 3 + biased reference |
| 3 | Implement feature distillation | Run Exp 1 (class labels ablation) |
| 4 | Run Exp 2 & 3 | Start CelebA teachers (if time) |
| 5 | Final experiments, analysis | Write blog post |

### Contingency Priorities

If time runs short:
1. **Must have**: AGRE-KD baseline + Exp 1 (class labels) on Waterbirds
2. **Should have**: Exp 2 (feature distillation) on Waterbirds
3. **Nice to have**: Exp 3 (combined) + CelebA experiments

---

## 8. Related Work

### Papers We Build On

1. **AGRE-KD** (Kenfack et al., 2024): Our direct baseline; adaptive gradient-based ensemble KD
2. **DFR** (Kirichenko et al., 2022): Last-layer retraining for debiasing; used to create debiased teachers
3. **FitNets** (Romero et al., 2015): Introduced intermediate feature distillation
4. **Debiasify** (2024): Self-distillation from deep to shallow layers for debiasing
5. **Yu et al. (2025)**: Showed layer-selection strategy has diminishing returns; one intermediate layer provides most benefit

### Key Insights from Literature

- **Penultimate layer contains decodable core features** even in biased models (Kirichenko et al.)
- Early/shallow layers predominantly encode spurious features (backgrounds, textures)
- Feature distillation can transfer inductive biases between architectures
- Debiased teachers can still produce biased students through distillation
- Class labels provide regularization that can prevent bias amplification
- Layer matching strategy matters less than having *some* intermediate supervision (Yu et al.)

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Colab disconnects during training | High | Checkpoint every 5 epochs to Google Drive |
| AGRE-KD replication fails | Medium | Fall back to simpler baselines (AVER, AE-KD) |
| Feature distillation implementation bugs | Medium | Test on small synthetic data first |
| CelebA training too slow | High | Focus on Waterbirds; subsample CelebA if needed |
| Negative results | Medium | Frame as "understanding limitations"; still publishable |

---

## 10. Questions for TA/Instructor Review

1. Is the scope appropriate for a 2-person, 5-day project?
2. Should we prioritize depth (more ablations on Waterbirds) or breadth (both datasets)?
3. Are there other baselines we should compare against?
4. Is the feature distillation approach (single layer, MSE loss) sufficiently novel?
5. If results are negative, what analysis would make the blog post compelling?

---

## Appendix A: Key Equations

**AGRE-KD Teacher Weighting**:
$$W_t(x_i) = 1 - \langle \nabla \ell_i^t(\theta), \nabla \ell_i^b(\theta) \rangle$$

**Weighted KD Loss**:
$$\mathcal{L}_{wKD} = \frac{W_t(x_i) \cdot \mathcal{L}_{KD}}{\sum_t W_t}$$

**Our Combined Loss**:
$$\mathcal{L}_{total} = (1-\alpha) \mathcal{L}_{cls} + \alpha \mathcal{L}_{wKD} + \gamma \mathcal{L}_{feat}$$

**Feature Distillation Loss** (penultimate layer):
$$\mathcal{L}_{feat} = \| \text{Adapter}(f_s^{(4)}) - f_t^{(4)} \|^2$$

Where $f^{(4)}$ denotes the layer4 (penultimate) features and Adapter is a 1×1 convolution aligning student dimensions (512) to teacher dimensions (2048).

---

## Appendix B: Dataset Statistics

### Waterbirds

| Group | Class | Background | Train | Test |
|-------|-------|------------|-------|------|
| 0 | Landbird | Land | 3,498 | 2,255 |
| 1 | Landbird | Water | 184 | 642 |
| 2 | Waterbird | Land | 56 | 133 |
| 3 | Waterbird | Water | 1,057 | 466 |

### CelebA (Blonde Hair)

| Group | Blonde | Gender | Train | Test |
|-------|--------|--------|-------|------|
| 0 | No | Female | 71,629 | 9,767 |
| 1 | No | Male | 66,874 | 7,535 |
| 2 | Yes | Female | 22,880 | 2,480 |
| 3 | Yes | Male | 1,387 | 182 |
