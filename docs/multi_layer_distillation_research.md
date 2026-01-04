# Multi-Layer Feature Distillation for Group-Robust Ensemble Knowledge Distillation

**Research Analysis & Implementation Plan**
**Date**: December 2025
**Project**: Extending AGRE-KD with Feature Distillation

---

## 1. Executive Summary

### Novelty Assessment

**Your feature distillation work IS novel.** The AGRE-KD paper (Kenfack et al., 2024) explicitly states:

> "we restrict ourselves to logit distillation and **leave feature distillation for future exploration**" — Page 3

The paper's Appendix C.1 already tested α < 1 (adding class labels), making that experiment less novel. However, **feature distillation remains unexplored** in the AGRE-KD context.

### Recommendation

**Progressive multi-layer feature distillation is a valid and feasible experimental direction** with:
- **Low implementation complexity** (~2-3 hours)
- **Expected gains**: +0.5-2% WGA over single-layer
- **High scientific value**: First exploration of multi-layer feature KD for group robustness

---

## 2. Background: What AGRE-KD Already Tested

### Paper's Main Contribution
- Gradient-based teacher weighting to improve worst-group accuracy (WGA)
- Uses biased model as reference to downweight teachers aligned with spurious correlations
- Focuses **only on logit distillation** (soft label matching)

### Appendix C.1: α Ablation (Already Tested)
The paper tested α ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} where:
- `L = α * L_KD + (1-α) * L_CE`
- Found WGA increases with α when teachers are debiased
- **Conclusion**: More weight on distillation (α→1) is better when teachers are robust

### What They Explicitly Left Out
From page 3:
> "Transferring knowledge from intermediary representation (feature-level) can provide more fine-grained information and boost the students' performance... On the other hand, recent studies have shown that models trained with ERM still learn core features while spurious features are only amplified in the last classifier layer. In this regard, we restrict ourselves to logit distillation and leave feature distillation for future exploration."

---

## 3. Current Experimental Results

### Single-Layer Feature Distillation (Your Work)

| Experiment | α | γ | WGA (%) | Δ from Baseline |
|------------|---|---|---------|-----------------|
| AGRE-KD Baseline | 1.0 | 0.0 | ~85.05 | — |
| + Feature Distillation | 1.0 | 0.25 | ~86.29 | +1.24% |
| AVER Baseline | 1.0 | 0.0 | ~83.33 | — |

**Current implementation**: Pooled features (after global average pooling)
- ResNet-18 student: 512 dimensions
- ResNet-50 teacher: 2048 dimensions
- Linear adapter to match dimensions

---

## 4. Research Support for Multi-Layer Feature Distillation

### 4.1 Shortcut Mitigation via Intermediate Layer KD (arXiv 2025)

**Paper**: [arXiv:2511.17421](https://arxiv.org/pdf/2511.17421)

**Key Finding**:
> "Different types of shortcuts—those that are diffuse and spread throughout the image, as well as those that are localized to specific areas—manifest distinctly across network layers and can therefore be more effectively targeted through mitigation strategies that target the intermediate layers."

**Relevance**: Spurious correlations (like backgrounds in Waterbirds) may be encoded differently at different network depths. Multi-layer distillation could help by:
1. Transferring robust semantic features from deep layers
2. Aligning spatial representations at mid-level layers
3. Avoiding spurious low-level features

### 4.2 Multi-Layer Feature Fusion KD (PLOS One 2023)

**Paper**: [PMC10461825](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10461825/)

**Methodology**:
- Extract features from layers 2, 3, 4 of VGG networks
- Apply Squeeze-and-Excitation blocks for attention
- Progressive fusion pyramid combining multi-resolution features

**Results**:
- ResNet20: +1.82% accuracy on CIFAR-100
- VGG8: +3.35% accuracy on CIFAR-100

**Key Insight**: "High-resolution features provide accurate localization while deeper features contain stronger semantic content."

### 4.3 Multistage Feature Fusion KD (Nature Scientific Reports 2024)

**Paper**: [nature.com/s41598-024-64041-4](https://www.nature.com/articles/s41598-024-64041-4)

**Approach**:
- Cross-stage feature fusion symmetric framework
- Attention mechanisms to enhance fused features
- Contrastive loss between teacher and student at same stage

**Problem Addressed**: "Significant differences in intermediate feature distributions between teacher and student models"

### 4.4 Progressive Knowledge Distillation (arXiv 2024)

**Paper**: [arXiv:2401.02677](https://arxiv.org/abs/2401.02677)

**Finding**: Layer-level losses are more granular than block-level losses, enabling:
- Identification of essential features at each depth
- Retention of critical representations during distillation

### 4.5 FitNets: Hints for Thin Deep Nets (Foundational Work)

**Paper**: [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)

**Original insight**: Using intermediate "hints" from teacher's hidden layers to guide student learning. Typically choose the middle layer as the hint layer.

---

## 5. Feasibility Analysis

### 5.1 Codebase Support (Already Implemented)

Your codebase already extracts all intermediate layers:

**`code/models.py` (lines 82-89)**:
```python
features = {
    'layer1': l1,      # 56×56 spatial, 64→256 channels
    'layer2': l2,      # 28×28 spatial, 128→512 channels
    'layer3': l3,      # 14×14 spatial, 256→1024 channels
    'layer4': l4,      # 7×7 spatial, 512→2048 channels
    'pooled': pooled,  # 512→2048 (after avgpool)
}
```

**`code/models.py` (lines 139-157)**: `create_feature_adapter()` handles dimension matching for all layers.

### 5.2 Layer Dimensions (ResNet-18 → ResNet-50)

| Layer | Student Channels | Teacher Channels | Spatial Size | Adapter |
|-------|------------------|------------------|--------------|---------|
| layer1 | 64 | 256 | 56×56 | Conv2d 1×1 |
| layer2 | 128 | 512 | 28×28 | Conv2d 1×1 |
| layer3 | 256 | 1024 | 14×14 | Conv2d 1×1 |
| layer4 | 512 | 2048 | 7×7 | Conv2d 1×1 |
| pooled | 512 | 2048 | — | Linear |

### 5.3 Implementation Effort

**Estimated time**: 2-3 hours

**Required changes**:
1. Add `MultiLayerFeatureDistillationLoss` class to `losses.py`
2. Update `train.py` to extract and pass multiple layers
3. Add per-layer gamma configuration to `config.py`

---

## 6. Important Caveat: DFR Teacher Similarity

### The Issue

Your DFR teachers are created by:
1. Training with ERM (standard training)
2. Retraining **only the last classification layer** on balanced data

This means:
- **All DFR teachers share identical backbone features**
- Only the classifier (last layer) differs
- Teacher feature averaging essentially returns the same features

### Implications

| Aspect | Impact |
|--------|--------|
| Multi-layer benefit | May be smaller than typical KD |
| Teacher weighting (AGRE) | Less meaningful for features (all same) |
| Student alignment | Still valuable at multiple depths |

### Why Multi-Layer Might Still Help

1. **Student representation alignment**: Even if teacher features are identical, forcing the student to match at multiple depths provides stronger supervision
2. **Gradient flow**: Multi-layer losses provide gradients to earlier student layers directly
3. **Regularization**: Matching at multiple scales acts as implicit regularization

---

## 7. Proposed Experiments

### Phase 1: Complete Gamma Ablation (Single-Layer Baseline)

| γ | Description | Purpose |
|---|-------------|---------|
| 0.0 | AGRE-KD baseline | Control |
| 0.05 | Weak feature distillation | Test sensitivity |
| 0.25 | Moderate (current best) | Baseline |
| 0.50 | Strong | Optimal search |
| 0.75 | Very strong | Optimal search |
| 1.00 | Equal weight (KD = features) | Upper bound |

### Phase 2: Progressive Multi-Layer Distillation

**Loss function**:
```
L_feat = γ2 * MSE(adapt(s_layer2), t_layer2)
       + γ3 * MSE(adapt(s_layer3), t_layer3)
       + γ4 * MSE(adapt(s_layer4), t_layer4)
```

**Experimental configurations**:

| Config | γ2 | γ3 | γ4 | Rationale |
|--------|----|----|----|----|
| A | 0.0 | 0.0 | 0.25 | Baseline (single-layer) |
| B | 0.0 | 0.1 | 0.25 | Add layer3 only |
| C | 0.05 | 0.1 | 0.25 | Add layer2 + layer3 |
| D | 0.1 | 0.2 | 0.3 | Increasing (deeper = more weight) |
| E | 0.2 | 0.15 | 0.1 | Decreasing (early = more weight) |

**Rationale for layer weights**:
- **Layer2**: Low-level (edges, textures) — may contain spurious features
- **Layer3**: Mid-level (object parts) — balanced features
- **Layer4**: High-level (semantics) — most robust features

---

## 8. Expected Outcomes

### Positive Outcome (Likely: 60%)
- Multi-layer provides +0.5-1.5% WGA over single-layer
- Config B or C performs best (adding layer3 helps)
- **Publishable claim**: "First exploration of multi-layer feature distillation for group-robust ensemble KD"

### Neutral Outcome (Possible: 30%)
- Multi-layer ≈ single-layer performance
- **Valuable finding**: For DFR teachers with shared backbones, layer selection doesn't matter
- Supports Kirichenko et al. insight that ERM features are already good

### Negative Outcome (Unlikely: 10%)
- Multi-layer hurts WGA
- **Insight**: Early layers transfer spurious features (avoid layer1/layer2)
- Recommendation to use only penultimate layer

---

## 9. Alternative Directions (If Multi-Layer Doesn't Help)

### 9.1 Contrastive Feature Distillation (CRD)
Replace MSE with InfoNCE loss:
```
L_CRD = -log(exp(s·t/τ) / Σ exp(s·t'/τ))
```
Benefits: Better for representation learning, handles modality gaps

### 9.2 Attention Transfer
Distill attention maps rather than raw features:
```
L_AT = ||A_s - A_t||_2  where A = sum(F^2, dim=channel)
```
Benefits: Focuses on "where" the model looks, not exact feature values

### 9.3 Per-Layer AGRE Weighting
Apply gradient-based weighting to feature loss per teacher:
```
W_t^layer = 1 - cos_sim(∇L_feat^t, ∇L_feat^biased)
```
Benefits: Uses AGRE insight at feature level

### 9.4 Feature Dropout
Randomly mask feature channels during distillation:
```
L_feat = MSE(dropout(s_feat), dropout(t_feat))
```
Benefits: Regularization, prevents overfitting to spurious patterns

---

## 10. Implementation Code

### MultiLayerFeatureDistillationLoss

```python
class MultiLayerFeatureDistillationLoss(nn.Module):
    """
    Multi-layer feature distillation with per-layer gamma weights.

    L_feat = Σ_l γ_l * MSE(adapt_l(student_l), teacher_l)
    """

    def __init__(self, layer_configs):
        """
        Args:
            layer_configs: Dict of {layer_name: {
                'gamma': float,
                'student_dim': int,
                'teacher_dim': int
            }}
        """
        super().__init__()
        self.adapters = nn.ModuleDict()
        self.gammas = {}

        for layer, cfg in layer_configs.items():
            if cfg['gamma'] > 0:
                spatial = (layer != 'pooled')
                if spatial:
                    self.adapters[layer] = nn.Conv2d(
                        cfg['student_dim'], cfg['teacher_dim'],
                        kernel_size=1, bias=False
                    )
                else:
                    self.adapters[layer] = nn.Linear(
                        cfg['student_dim'], cfg['teacher_dim'], bias=False
                    )
                self.gammas[layer] = cfg['gamma']

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Dict {layer_name: tensor}
            teacher_features: Dict {layer_name: tensor}

        Returns:
            total_loss: Combined multi-layer feature loss
            losses: Dict with per-layer losses for logging
        """
        total_loss = 0.0
        losses = {}

        for layer, gamma in self.gammas.items():
            s_feat = self.adapters[layer](student_features[layer])
            t_feat = teacher_features[layer].detach()

            # Handle spatial size mismatch (ResNet-18/50 have same spatial dims, but just in case)
            if s_feat.dim() == 4 and s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])

            layer_loss = F.mse_loss(s_feat, t_feat)
            losses[f'feat_{layer}'] = layer_loss.item()
            total_loss += gamma * layer_loss

        return total_loss, losses
```

---

## 11. References

1. **AGRE-KD**: Kenfack et al. (2024). "Adaptive Group Robust Ensemble Knowledge Distillation." TMLR. [arXiv:2411.14984](https://arxiv.org/abs/2411.14984)

2. **DFR**: Kirichenko et al. (2022). "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations." ICLR.

3. **FitNets**: Romero et al. (2014). "FitNets: Hints for Thin Deep Nets." [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)

4. **Multi-Layer Feature Fusion KD**: Wang et al. (2023). PLOS One. [PMC10461825](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10461825/)

5. **Multistage Feature Fusion KD**: Chen et al. (2024). Nature Scientific Reports. [s41598-024-64041-4](https://www.nature.com/articles/s41598-024-64041-4)

6. **Shortcut Mitigation via Intermediate Layer KD**: Boland et al. (2025). [arXiv:2511.17421](https://arxiv.org/pdf/2511.17421)

7. **Progressive Layer-Level KD**: Kim et al. (2024). [arXiv:2401.02677](https://arxiv.org/abs/2401.02677)

---

## 12. Summary: Recommended Path Forward

1. **Complete gamma ablation** — Establish optimal single-layer γ*
2. **Implement MultiLayerFeatureDistillationLoss** — ~2-3 hours
3. **Run progressive experiments** — Configs B, C, D (3-5 runs)
4. **Analyze layer contributions** — Which layers help WGA most?
5. **Document findings** — Novel contribution extending AGRE-KD
