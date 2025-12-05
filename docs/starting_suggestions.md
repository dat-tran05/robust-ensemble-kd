# Comprehensive Project Plan for Robust Ensemble Knowledge Distillation

Your 5-day project to extend AGRE-KD with class labels and feature distillation is **feasible but tight**—prioritize Waterbirds over CelebA and use pretrained weights. No public AGRE-KD code exists yet, but you can build on DFR and AE-KD repositories. Feature distillation via penultimate layer MSE with a simple adapter is implementable in <50 lines and may improve group robustness based on recent research showing earlier layers encode spurious correlations.

## Code repositories and starting points

**Critical finding**: AGRE-KD (arxiv 2411.14984) has **no public code** as of December 2025—you'll need to implement from the paper or contact authors. However, several excellent alternatives exist:

For **datasets and group robustness**, use **WILDS** (`pip install wilds`) which provides standardized PyTorch dataloaders with automatic downloads for Waterbirds and CelebA. The **Group DRO repository** (github.com/kohpangwei/group_DRO, 286 stars, MIT license) contains dataloaders in `data/data.py` plus the script `generate_waterbirds.py` for regenerating the dataset.

For **debiasing methods**, the **DFR repository** (github.com/PolinaKirichenko/deep_feature_reweighting, 110 stars) includes `train_classifier.py`, CelebA/Waterbirds loaders in `wb_data.py`, and pre-trained checkpoints on Google Drive. The **JTT repository** (github.com/anniesch/jtt) provides the two-stage upweighting approach.

For **ensemble KD baselines**, use **AE-KD** (github.com/AnTuo1998/AE-KD) which implements Adaptive Ensemble KD and includes AVER baseline—run via `python train_multiTeacher.py --ensemble_method AEKD`. The comprehensive **torchdistill** framework (github.com/yoshitomo-matsubara/torchdistill, 1.3k stars) offers 26+ KD methods including FitNets and Attention Transfer with YAML configuration.

| Repository | Purpose | Stars |
|------------|---------|-------|
| p-lambda/wilds | Dataset loaders | 586 |
| PolinaKirichenko/deep_feature_reweighting | DFR method | 110 |
| AnTuo1998/AE-KD | Ensemble KD | ~50 |
| yoshitomo-matsubara/torchdistill | KD framework | 1,300+ |

## Feature distillation implementation in under 50 lines

**FitNets** extends KD to intermediate representations using a regressor to match dimensions between student and teacher layers. The loss is simply `L_hint = ||W_r × h_student - h_teacher||²` where `W_r` is typically a 1×1 convolution. Training proceeds in two stages: first train the guided layer to match the hint, then train the full network.

**Attention Transfer** is even simpler—no architectural changes required. Compute spatial attention maps by summing squared activations over channels: `Q = Σ|A_i|²`, then match L2-normalized attention maps between student and teacher. The loss formula is `L_AT = β × ||Q_S/||Q_S|| - Q_T/||Q_T||||²` with β≈1000 for CIFAR-scale problems.

```python
# Complete feature distillation loss (~20 lines)
class FeatureDistillLoss(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.adapter = nn.Conv2d(student_dim, teacher_dim, 1) if student_dim != teacher_dim else nn.Identity()
    
    def forward(self, s_feat, t_feat):
        s_feat = self.adapter(s_feat)
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
        return F.mse_loss(s_feat, t_feat.detach())
```

**Which layer to use**: For ResNet-18, extract from **layer2** or **layer3** (middle residual blocks). Recent research on intermediate layer classifiers shows that **earlier layers generalize better for out-of-distribution data**—layer 5-6 of ResNet-18 achieved 80.6% WGA on CelebA versus lower WGA from penultimate layers. This suggests feature distillation from earlier layers may specifically help group robustness.

## Compute feasibility is tight but achievable

The T4 GPU (16GB VRAM, 2,560 CUDA cores) runs at roughly **3.6× slower than V100** for ResNet training. Key timing estimates:

| Task | Waterbirds | CelebA |
|------|------------|--------|
| ResNet-50 per epoch | 1-2 min | 8-15 min |
| Full training (50-100 epochs) | 1-3 hrs | 6-12 hrs |
| 3 teachers | 3-9 hrs | 18-36 hrs |
| Student distillation (×3 configs) | 2-4 hrs | 8-24 hrs |

**Total estimate**: Waterbirds experiments need ~15-25 hours; CelebA needs ~40-80 hours. With 2 people × 5 days × 12-24 GPU-hours = **120-240 GPU-hours available**, this is feasible but requires **using pretrained ImageNet weights** (non-negotiable—saves 50-80% time) and careful prioritization.

**Google Colab Pro constraints**: Sessions last up to 12 hours (24 for Pro+) with idle timeouts and unpublished usage limits. Checkpoint to Google Drive every 5-10 epochs, alternate team member accounts, and close tabs when not in use.

**Recommended timeline**:
- **Day 1**: Setup + train 3 Waterbirds teachers (parallel on 2 accounts)
- **Day 2**: Waterbirds distillation experiments + start CelebA teachers
- **Day 3**: Complete CelebA teachers + Waterbirds ablations
- **Day 4**: CelebA distillation + hyperparameter sweeps
- **Day 5**: Final experiments + analysis + write-up (buffer for failures)

## Combined loss function structure for your experiments

Your three experiments map directly to this unified loss formulation:

```
L_total = (1-α) × L_CE + α × L_KD + γ × L_feature
```

**Experiment 1** (class labels, α < 1): Set α=0.7-0.9, γ=0. This adds ground-truth supervision alongside distillation—the original AGRE-KD uses α=1 (no class labels).

**Experiment 2** (feature distillation only, α=1): Set α=1, γ=0.1-0.25. Use MSE between adapted student features and teacher features at layer2 or layer3.

**Experiment 3** (combined): Set α=0.7-0.9, γ=0.1-0.25. This combines all three signals.

```python
def combined_distillation_loss(s_logits, t_logits, s_feat, t_feat, labels, T=4, alpha=0.7, gamma=0.1):
    # Ground truth loss
    ce = F.cross_entropy(s_logits, labels)
    # Soft label KD loss
    kd = F.kl_div(F.log_softmax(s_logits/T, dim=1), 
                  F.softmax(t_logits.detach()/T, dim=1), 
                  reduction='batchmean') * (T*T)
    # Feature matching (assumes adapter applied)
    feat = F.mse_loss(s_feat, t_feat.detach())
    return (1-alpha)*ce + alpha*kd + gamma*feat
```

## Hyperparameters with highest impact

**Temperature (τ)**: Start with τ=4, optionally sweep τ∈{2, 4, 10}. Higher temperatures reveal inter-class relationships ("dark knowledge"); τ=4-10 works across most tasks. Lower τ when student is much smaller than teacher.

**Alpha (α)**: The original KD paper uses α=0.1 (high distillation weight). For your experiments comparing α=1 vs α<1, try α∈{0.5, 0.7, 0.9}. Lower α means more reliance on teachers.

**Feature loss weight (γ)**: Start γ=0.25; the Debiasify paper uses γ=0.1 for debiasing. Feature loss is typically smaller scale than logit loss.

**Training**: Use SGD with lr=0.01, momentum=0.9, weight_decay=1e-4; batch size 32 for Waterbirds, 64-128 for CelebA. For distillation, 30-50 epochs often suffice with early stopping on validation WGA.

## Evaluation metrics beyond WGA and average accuracy

**Primary**: Worst-Group Accuracy (WGA) = min_{g∈G} Accuracy(g), the standard metric for spurious correlation robustness. Report **per-group accuracy breakdown** (4 values for each dataset) to show which groups improve.

**Secondary metrics**:
- **Accuracy gap**: max_g Accuracy(g) - min_g Accuracy(g)
- **Expected Calibration Error (ECE)**: though research shows ECE doesn't correlate strongly with robustness
- **Fairness metrics**: Equalized Odds (equal TPR/FPR across groups) or Equal Opportunity (equal TPR only)

## Baseline results you need to beat

**Waterbirds** (4,795 training images, hardest group: waterbird-on-land with only 56 samples):

| Method | WGA | Avg Acc |
|--------|-----|---------|
| ERM | 68-72% | 97% |
| JTT | 86-87% | 93% |
| DFR | 91-93% | 94% |
| Group DRO | 90-91% | 93% |

**CelebA blonde hair** (162k images, hardest group: blond males with 1% of data):

| Method | WGA | Avg Acc |
|--------|-----|---------|
| ERM | 44-47% | 95% |
| JTT | 76-81% | 88% |
| DFR | 88-89% | 91% |
| Group DRO | 87-89% | 93% |

**For a class project**: Matching JTT (~85% Waterbirds, ~80% CelebA) is "good"; matching DFR (~91%, ~88%) is "very good." Report mean±std over 3 seeds when possible.

## Recent research suggests feature distillation may help debiasing

**Debiasify (November 2024)** found that shallow layers encode spurious correlations while deep layers contain predictive features. Their method uses self-distillation from layer2→layer4 to align distributions, achieving **10% WGA improvement on CelebA**. This directly supports your hypothesis that feature distillation could improve group robustness.

**Intermediate Layer Classifiers (2025)** showed that middle layers of ResNet-18 achieve better worst-group accuracy than penultimate layers on CelebA—**layer 5-6 reached 80.6% WGA** while later layers performed worse. This suggests distilling from earlier layers may specifically help.

**DeTT (2023)** found that even debiased teachers produce biased students when trained on biased data—they recommend transplanting the last layer and matching penultimate features, not just logits.

These papers suggest your Experiment 2 (feature distillation at intermediate layers) has theoretical grounding, and combining with class labels (Experiment 3) could outperform logit-only AGRE-KD.

## Practical ensemble KD implementation tips

**Handle multiple teachers efficiently**: Pre-compute teacher logits for the entire training set and save to disk. This avoids loading all teachers during training:

```python
# Pre-compute once
teacher_logits = {}
for name, teacher in teachers.items():
    teacher.eval()
    with torch.no_grad():
        teacher_logits[name] = [teacher(batch) for batch in loader]
# Average during training
ensemble_logits = torch.stack([teacher_logits[n][batch_idx] for n in teachers]).mean(0)
```

**Memory**: ResNet-50 is ~100MB per model; 3-5 teachers use 300-500MB. T4's 16GB is sufficient, but reduce batch size if needed. Use `torch.no_grad()` for teacher forward passes.

**Checkpointing**: Save to Google Drive every 5-10 epochs with optimizer state for resumption after Colab disconnects:
```python
torch.save({'epoch': e, 'model': model.state_dict(), 'opt': opt.state_dict(), 'best_wga': best_wga}, 
           '/content/drive/MyDrive/ckpt.pt')
```

## Blog post structure for your results

Whether results are positive, negative, or mixed, structure your report following Stanford CS229 guidelines (~5 pages):

1. **Abstract** (100 words): Motivation, method, results, key insight
2. **Introduction** (0.5 pages): Why group robustness matters, your research questions
3. **Background** (0.5 pages): AGRE-KD, feature distillation, datasets
4. **Methods** (0.75 pages): Your three loss formulations with equations
5. **Experiments** (1.5 pages): Main results table, ablations, per-group analysis
6. **Discussion** (0.5 pages): Why things worked/didn't, when to use this approach
7. **Conclusion** (0.25 pages): Takeaways, future work

**For negative results**: Frame as "understanding when X works"—the ICML 2024 position paper on negative results emphasizes that documenting what doesn't work is scientifically valuable. Explain **why** it didn't work, characterize boundary conditions, and suggest future directions.

**Tables**: Include main results comparing all methods × metrics, plus ablation showing contribution of class labels alone, features alone, and combined. Visualize per-group performance as a bar chart.

**Code**: Include GitHub link with training scripts and README—especially important for negative results to enable verification and prevent others from repeating failed approaches.

## Recommended experiment priority if time runs short

1. **Essential**: 3 teachers + baseline ERM + AGRE-KD replication on Waterbirds
2. **Core**: Your 3 experiments on Waterbirds with α and γ ablations
3. **If time permits**: Repeat on CelebA (consider 50% subsample)
4. **Optional**: Temperature sweep, different layer choices for feature distillation

Start with 3 teachers (not 5) to save 40% training time, and subsample CelebA to 50% if Day 4 arrives without CelebA teacher training complete. The key insight—whether class labels and/or feature distillation improve ensemble KD for group robustness—can be demonstrated convincingly on Waterbirds alone.