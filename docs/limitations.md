# Limitations and Implementation Notes

## 1. WGA Gap vs. Original Paper

Our AGRE-KD baseline achieves **84.48 ± 1.02% WGA**, slightly lower than the paper's reported results (~87.9% with 10 teachers, >86% with 5 teachers per Figure 7).

### Potential Causes

1. **Training Hyperparameters**
   - Our setup: 5 teachers, 30 epochs, lr=0.001, batch_size=128
   - Paper may use different learning rate, epochs, or weight decay

2. **Ensemble Size**
   - Paper uses 10 teachers by default
   - Our experiments use 5 teachers
   - Paper's Figure 7 results at ensemble size 5 may use different seeds/setup

3. **Random Variance**
   - Paper reports 87.9 ± 1.23% (significant variance)
   - Our 84.48 ± 1.02% could reflect seed/initialization differences

### Verification Performed

- **Biased model check**: Verified biased reference model has WGA = 73.8% (correct - not accidentally debiased)
- **Teacher quality**: Our DFR-debiased teachers achieve 91.9-93.8% WGA (comparable to paper's ~90.9%)

### Impact on Our Findings

This gap does **not** invalidate our feature distillation findings because:
- All comparisons (γ=0 vs γ=0.5 vs γ=1.0) use the **same experimental setup**
- Relative improvements/changes are meaningful
- The conclusion that feature distillation provides marginal benefit remains valid

---

## 2. Reproducibility Notes

- Seeds 42, 43, 44 used for 3-seed statistical significance
- Original "og" runs (no seed set) excluded from statistical analysis
- All experiments use identical hyperparameters except the variable being tested

---

## 3. Suggested Blog Citation

> "Our AGRE-KD baseline achieves 84.48 ± 1.02% WGA, slightly lower than the paper's reported results. This difference may stem from hyperparameter choices (we use 5 teachers, 30 epochs, lr=0.001) or random seed variance. Importantly, all our comparative experiments use identical setups, so relative findings remain valid."
