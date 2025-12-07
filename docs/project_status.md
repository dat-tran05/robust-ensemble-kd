# Project Status: Robust Ensemble Knowledge Distillation

**Last Updated**: December 7, 2025
**Deadline**: December 9, 2025 (11:59 PM)
**Team Size**: 2 members

---

## 1. Executive Summary

We are investigating whether feature distillation can improve AGRE-KD (Adaptive Group Robust Ensemble Knowledge Distillation) for group robustness. Our experiments so far show **marginal, non-statistically-significant improvement** from adding feature distillation (γ > 0), which constitutes a valid "negative result" per project guidelines.

**Key Finding**: Feature distillation provides limited benefit to AGRE-KD, likely because DFR-debiased teachers only debias the classifier head—the backbone features remain biased.

---

## 2. Completed Work

### 2.1 Infrastructure
- [x] GitHub repository setup (`dat-tran05/robust-ensemble-kd`)
- [x] Google Colab notebooks for experiments
- [x] Checkpointing system (resume from disconnects)
- [x] Results logging to JSON

### 2.2 Teacher Training
- [x] 5 ERM teachers (ResNet-50, seeds 42-46)
- [x] DFR debiasing applied to all teachers (~90% WGA each)
- [x] Biased reference model (ERM, seed 42)

### 2.3 Baseline Experiments (Completed)
| Experiment | α | γ | WGA (Mean ± Std) | n | Status |
|------------|---|---|------------------|---|--------|
| baseline_agrekd | 1.0 | 0.0 | 85.10 ± 0.45% | 3 | Done |
| exp2_gamma025 | 1.0 | 0.25 | 85.36 ± 0.92% | 3 | Done |
| aver_baseline | 1.0 | 0.0 | 84.27 ± 0.71% | 3 | Done |
| exp3_a09_g025 | 0.9 | 0.25 | 83.18% | 1 | Incomplete (running) |

### 2.4 Key Observations from Initial Experiments
1. **Feature distillation (γ=0.25)**: +0.26% over baseline — NOT statistically significant
2. **Adding class labels (α < 1)**: HURTS performance (exp3 < baseline)
3. **AVER vs AGRE-KD**: AGRE-KD is better (+0.83%)

---

## 3. Key Decisions Made

### 3.1 Experimental Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Temperature (τ)** | 4.0 | AGRE-KD paper used this; their ablation shows it's optimal |
| **Feature layer** | Penultimate (layer4) | DFR operates here; core features are decodable here |
| **Skip α ablation** | No more α values | Paper shows α=1.0 optimal; our results confirm |
| **Add gamma ablation** | γ ∈ {0.05, 0.50, 0.75, 1.0} | Shows full sensitivity curve |
| **Skip AE-KD** | Not implementing | Would require new code; paper already compares |
| **Skip temperature ablation** | Not running | Paper already did this (Figure 5) |
| **3 seeds per experiment** | Statistical significance | Standard in AGRE-KD paper |

### 3.2 Narrative Decision

**Framing**: "Negative result with analysis"
The AGRE-KD paper explicitly states: "We leave feature distillation for future exploration." We directly investigated this and found limited benefit—a meaningful contribution that explains WHY this approach doesn't help.

**Core insight**: DFR only debiases the classifier head; backbone features remain biased. Feature distillation transfers these biased features to the student.

---

## 4. Remaining Work

### 4.1 Experiments (Tonight - Overnight)

**Computer 1** (~4 hrs):
| Experiment | α | γ | Seed | Status |
|------------|---|---|------|--------|
| exp3_a09_g025 | 0.9 | 0.25 | 43 | Pending |
| exp3_a09_g025 | 0.9 | 0.25 | 44 | Pending |
| gamma_075 | 1.0 | 0.75 | 42 | Pending |
| gamma_100 | 1.0 | 1.00 | 42 | Pending |

**Computer 2** (~3 hrs):
| Experiment | α | γ | Seed | Status |
|------------|---|---|------|--------|
| gamma_005 | 1.0 | 0.05 | 42 | Pending |
| gamma_050 | 1.0 | 0.50 | 42 | Pending |
| aver_gamma025 | 1.0 | 0.25 | 42 | Pending |

### 4.2 Required Figures (4-6 for blog)

| # | Figure | Data Source | Status |
|---|--------|-------------|--------|
| 1 | Main WGA comparison (bar chart with error bars) | Final results table | Pending |
| 2 | Gamma sensitivity curve (γ vs WGA) | Tonight's experiments | Pending |
| 3 | Per-group accuracy breakdown (4 groups × methods) | Existing + new results | Pending |
| 4 | AGRE vs AVER comparison | exp2 vs aver_gamma025 | Pending |
| 5 | Training dynamics (optional) | Training logs | Pending |
| 6 | Architecture diagram (optional) | Manual creation | Pending |

### 4.3 Blog Outline

**Title**: "Investigating Feature Distillation in Group-Robust Knowledge Distillation"

| Section | Words | Content |
|---------|-------|---------|
| 1. Introduction | 300 | Spurious correlations, KD, AGRE-KD, our question |
| 2. Background | 400 | Waterbirds, DFR, AGRE-KD methodology |
| 3. Method | 400 | Three extensions (feature, labels, combined) |
| 4. Experiments | 500 | Setup, main results table, gamma ablation |
| 5. Analysis | 600 | Why feature distillation has limited benefit |
| 6. Conclusion | 200 | Negative result insight, future work |
| **Total** | ~2400 | + 4-6 figures |

---

## 5. Timeline (Remaining 2 Days)

### December 7 (Tonight)
- [ ] Push notebook changes to GitHub
- [ ] Start overnight experiments (both computers)
- [ ] exp3 completing on current run

### December 8 (Tomorrow)
- [ ] Morning: Check overnight results
- [ ] Morning: Compile final results table
- [ ] Afternoon: Create all figures
- [ ] Afternoon: Statistical analysis (t-tests, CIs)
- [ ] Evening: Draft blog sections (split between team)

### December 9 (Deadline)
- [ ] Morning: Combine drafts, polish analysis
- [ ] Afternoon: Final review and formatting
- [ ] Submit before 11:59 PM

---

## 6. Expected Final Results Table

After all experiments complete:

| Method | α | γ | WGA (%) | Δ vs Baseline |
|--------|---|---|---------|---------------|
| AVER baseline | 1.0 | 0.0 | 84.27 ± 0.71 | -0.83 |
| **AGRE-KD (baseline)** | 1.0 | 0.0 | **85.10 ± 0.45** | — |
| AGRE-KD + γ=0.05 | 1.0 | 0.05 | ? | ? |
| AGRE-KD + γ=0.25 | 1.0 | 0.25 | 85.36 ± 0.92 | +0.26 |
| AGRE-KD + γ=0.50 | 1.0 | 0.50 | ? | ? |
| AGRE-KD + γ=0.75 | 1.0 | 0.75 | ? | ? |
| AGRE-KD + γ=1.00 | 1.0 | 1.00 | ? | ? |
| AVER + γ=0.25 | 1.0 | 0.25 | ? | ? |
| Combined (α=0.9, γ=0.25) | 0.9 | 0.25 | ? | ? |

---

## 7. Gamma Sensitivity Curve (Expected)

```
WGA (%)
 86 |
 85 |----*----*----*----?----?----?
 84 |
 83 |
    +----+----+----+----+----+----+
    0   0.05 0.25 0.50 0.75 1.00
                  γ
```

**Expected pattern**: Marginal improvement up to γ≈0.25, then plateau or decline.

---

## 8. Key Files

### Notebooks
- `light-code/notebooks/02_experiments.ipynb` — Original experiments
- `light-code/notebooks/03_seed_experiments.ipynb` — Multi-seed + gamma ablation

### Core Code
- `light-code/train.py` — Training loop with feature distillation
- `light-code/losses.py` — AGREKDLoss, FeatureDistillationLoss
- `light-code/config.py` — Experiment configurations

### Results
- `logs/seed_experiment_results.json` — All experiment results
- `checkpoints/` — Model checkpoints

---

## 9. Contingency Plans

### If overnight experiments fail
- Use existing 3-seed data for main conclusions
- Mark gamma ablation as "preliminary" with n=1

### If feature distillation shows NO improvement
- Frame as valid negative result (per project guidelines)
- Emphasize insight: DFR teachers don't benefit from feature distillation because features remain biased

### If time runs short
- Prioritize: Main results table + 4 core figures
- Cut: Detailed per-epoch analysis, extensive ablations
- Keep: Statistical significance testing (essential)

---

## 10. References

1. **AGRE-KD** (Kenfack et al., 2025): Our baseline method
2. **DFR** (Kirichenko et al., 2022): Teacher debiasing
3. **Waterbirds** (Sagawa et al., 2019): Dataset
4. **Feature Distillation** (Romero et al., 2015): FitNets
5. **Project Guidelines**: 6.7960 Final Project (2000-3000 words, 4-6 figures)
