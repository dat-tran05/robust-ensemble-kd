"""
Loss functions for knowledge distillation experiments.

Implements:
1. Standard KD loss (soft label matching)
2. Feature distillation loss (MSE on penultimate features)
3. Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# KNOWLEDGE DISTILLATION LOSS
# =============================================================================

class KDLoss(nn.Module):
    """
    Standard Knowledge Distillation loss using KL divergence.
    
    L_KD = KL(softmax(z_s/T) || softmax(z_t/T)) * T^2
    
    Args:
        temperature: Softmax temperature (higher = softer distributions)
    """
    
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: Student predictions [B, C]
            teacher_logits: Teacher predictions [B, C] (will be detached)
        """
        T = self.temperature
        
        # Soft targets from teacher (detached)
        soft_targets = F.softmax(teacher_logits.detach() / T, dim=1)
        
        # Log-softmax of student predictions
        log_soft_student = F.log_softmax(student_logits / T, dim=1)
        
        # KL divergence, scaled by T^2
        loss = F.kl_div(log_soft_student, soft_targets, reduction='batchmean') * (T * T)
        
        return loss


# =============================================================================
# FEATURE DISTILLATION LOSS
# =============================================================================

class FeatureDistillationLoss(nn.Module):
    """
    Feature distillation loss using MSE on intermediate representations.
    
    L_feat = MSE(adapter(f_student), f_teacher)
    
    Optionally handles spatial mismatches via adaptive pooling.
    
    Args:
        student_dim: Student feature dimension
        teacher_dim: Teacher feature dimension  
        spatial: If True, features are spatial (use Conv2d adapter)
    """
    
    def __init__(self, student_dim=512, teacher_dim=2048, spatial=False):
        super().__init__()
        
        # Dimension adapter
        if student_dim == teacher_dim:
            self.adapter = nn.Identity()
        elif spatial:
            self.adapter = nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False)
        else:
            self.adapter = nn.Linear(student_dim, teacher_dim, bias=False)
        
        self.spatial = spatial
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Student features [B, D_s] or [B, C_s, H, W]
            teacher_features: Teacher features [B, D_t] or [B, C_t, H, W]
        """
        # Adapt student dimensions
        s_adapted = self.adapter(student_features)
        
        # Handle spatial size mismatch (if spatial features)
        if self.spatial and s_adapted.shape[-2:] != teacher_features.shape[-2:]:
            s_adapted = F.adaptive_avg_pool2d(s_adapted, teacher_features.shape[-2:])
        
        # MSE loss (teacher is detached)
        loss = F.mse_loss(s_adapted, teacher_features.detach())

        return loss


# =============================================================================
# DISAGREEMENT-WEIGHTED FEATURE DISTILLATION LOSS
# =============================================================================

class DisagreementWeightedFeatureLoss(nn.Module):
    """
    Feature distillation that weights dimensions by teacher agreement.

    High agreement (low variance across teachers) = high weight.
    High disagreement (high variance) = low weight.

    Rationale: Teachers trained with different seeds may learn different
    spurious features. Where they agree → likely robust features.
    Where they disagree → likely spurious features.

    Args:
        student_dim: Student feature dimension
        teacher_dim: Teacher feature dimension
        eps: Small constant for numerical stability
    """

    def __init__(self, student_dim=512, teacher_dim=2048, eps=1e-6):
        super().__init__()
        if student_dim != teacher_dim:
            self.adapter = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.adapter = nn.Identity()
        self.eps = eps

    def forward(self, student_features, teacher_features_list):
        """
        Compute disagreement-weighted feature distillation loss.

        Args:
            student_features: Student pooled features [B, D_s]
            teacher_features_list: List of teacher features, each [B, D_t]

        Returns:
            loss: Scalar weighted MSE loss
        """
        # Adapt student to teacher dimension
        s_feat = self.adapter(student_features)  # [B, D_t]

        # Stack teachers: [T, B, D]
        stacked = torch.stack(teacher_features_list, dim=0)

        # Compute mean and variance across teachers
        avg_teacher = stacked.mean(dim=0)  # [B, D]
        variance = stacked.var(dim=0)  # [B, D]

        # Confidence weights: inverse variance (high agreement = high weight)
        confidence = 1.0 / (variance + self.eps)  # [B, D]

        # Normalize per sample to mean=1 (so total loss magnitude is similar)
        confidence = confidence / confidence.mean(dim=-1, keepdim=True)

        # Weighted MSE loss
        diff = s_feat - avg_teacher.detach()
        loss = (confidence.detach() * diff.pow(2)).mean()

        return loss


# =============================================================================
# MULTI-LAYER FEATURE DISTILLATION LOSS
# =============================================================================

class MultiLayerFeatureDistillationLoss(nn.Module):
    """
    Multi-layer feature distillation with per-layer gamma weights.

    L_feat = Σ_l γ_l * MSE(adapt_l(student_l), teacher_l)

    Supports layers: layer2, layer3, layer4, pooled

    Args:
        layer_gammas: Dict of {layer_name: gamma} for each layer to distill
        student_arch: Student architecture ('resnet18')
        teacher_arch: Teacher architecture ('resnet50')
    """

    # Feature dimensions by architecture and layer
    DIMS = {
        'resnet18': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'pooled': 512},
        'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'pooled': 2048},
    }

    def __init__(self, layer_gammas, student_arch='resnet18', teacher_arch='resnet50'):
        super().__init__()

        self.adapters = nn.ModuleDict()
        self.gammas = {}

        student_dims = self.DIMS[student_arch]
        teacher_dims = self.DIMS[teacher_arch]

        for layer, gamma in layer_gammas.items():
            if gamma > 0:
                student_dim = student_dims[layer]
                teacher_dim = teacher_dims[layer]
                spatial = (layer != 'pooled')

                if spatial:
                    self.adapters[layer] = nn.Conv2d(
                        student_dim, teacher_dim,
                        kernel_size=1, bias=False
                    )
                else:
                    self.adapters[layer] = nn.Linear(
                        student_dim, teacher_dim, bias=False
                    )
                self.gammas[layer] = gamma

        # Store total gamma for compatibility
        self.total_gamma = sum(self.gammas.values())

    def forward(self, student_features, teacher_features):
        """
        Compute multi-layer feature distillation loss.

        Args:
            student_features: Dict {layer_name: tensor} from student
            teacher_features: Dict {layer_name: tensor} from teacher

        Returns:
            total_loss: Combined multi-layer feature loss
            losses: Dict with per-layer losses for logging
        """
        total_loss = 0.0
        losses = {}

        for layer, gamma in self.gammas.items():
            s_feat = self.adapters[layer](student_features[layer])
            t_feat = teacher_features[layer].detach()

            # Handle spatial size mismatch (shouldn't happen for same-family ResNets)
            if s_feat.dim() == 4 and s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])

            layer_loss = F.mse_loss(s_feat, t_feat)
            losses[f'feat_{layer}'] = layer_loss.item()
            total_loss += gamma * layer_loss

        losses['feat_total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, losses


# =============================================================================
# COMBINED DISTILLATION LOSS
# =============================================================================

class CombinedDistillationLoss(nn.Module):
    """
    Combined loss for your three experiments:
    
    L_total = (1 - alpha) * L_CE + alpha * L_KD + gamma * L_feat
    
    Experiment 1: alpha < 1, gamma = 0 (add class labels)
    Experiment 2: alpha = 1, gamma > 0 (add feature distillation)
    Experiment 3: alpha < 1, gamma > 0 (combined)
    
    Args:
        alpha: Weight for KD vs CE (1.0 = pure KD, 0.0 = pure CE)
        gamma: Weight for feature distillation loss
        temperature: Temperature for KD loss
        student_dim: Student feature dimension (for adapter)
        teacher_dim: Teacher feature dimension
    """
    
    def __init__(self, alpha=1.0, gamma=0.0, temperature=4.0,
                 student_dim=512, teacher_dim=2048):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        
        # Component losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(temperature=temperature)
        
        # Feature loss (only create if gamma > 0)
        if gamma > 0:
            self.feat_loss = FeatureDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                spatial=False  # Use pooled features
            )
        else:
            self.feat_loss = None
    
    def forward(self, student_logits, teacher_logits, labels,
                student_features=None, teacher_features=None):
        """
        Compute combined loss.
        
        Args:
            student_logits: Student predictions [B, C]
            teacher_logits: Teacher predictions [B, C]
            labels: Ground truth labels [B]
            student_features: Student features [B, D] (optional, for gamma > 0)
            teacher_features: Teacher features [B, D] (optional, for gamma > 0)
            
        Returns:
            loss: Combined loss
            loss_dict: Dict with component losses for logging
        """
        losses = {}
        total_loss = 0.0
        
        # Cross-entropy with ground truth
        if self.alpha < 1.0:
            ce = self.ce_loss(student_logits, labels)
            losses['ce'] = ce.item()
            total_loss += (1 - self.alpha) * ce
        
        # Knowledge distillation
        kd = self.kd_loss(student_logits, teacher_logits)
        losses['kd'] = kd.item()
        total_loss += self.alpha * kd
        
        # Feature distillation
        if self.gamma > 0 and self.feat_loss is not None:
            if student_features is None or teacher_features is None:
                raise ValueError("Features required when gamma > 0")
            
            feat = self.feat_loss(student_features, teacher_features)
            losses['feat'] = feat.item()
            total_loss += self.gamma * feat
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


# =============================================================================
# ENSEMBLE KD LOSS (for multiple teachers)
# =============================================================================

class EnsembleKDLoss(nn.Module):
    """
    Ensemble KD loss that averages soft targets from multiple teachers.
    
    For AGRE-KD, you would modify this to use weighted averaging.
    This is the simpler baseline (AVER - simple averaging).
    
    Args:
        alpha: Weight for KD vs CE
        gamma: Weight for feature distillation
        temperature: KD temperature
    """
    
    def __init__(self, alpha=1.0, gamma=0.0, temperature=4.0,
                 student_dim=512, teacher_dim=2048):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        
        if gamma > 0:
            self.feat_loss = FeatureDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                spatial=False
            )
        else:
            self.feat_loss = None
    
    def forward(self, student_logits, teacher_logits_list, labels,
                student_features=None, teacher_features_list=None):
        """
        Args:
            student_logits: Student predictions [B, C]
            teacher_logits_list: List of teacher predictions, each [B, C]
            labels: Ground truth [B]
            student_features: Student features [B, D]
            teacher_features_list: List of teacher features, each [B, D]
        """
        losses = {}
        total_loss = 0.0
        T = self.temperature
        
        # Cross-entropy
        if self.alpha < 1.0:
            ce = self.ce_loss(student_logits, labels)
            losses['ce'] = ce.item()
            total_loss += (1 - self.alpha) * ce
        
        # Average teacher logits (simple ensemble)
        avg_teacher_logits = torch.stack(teacher_logits_list, dim=0).mean(dim=0)
        
        # KD loss with averaged teacher
        soft_targets = F.softmax(avg_teacher_logits.detach() / T, dim=1)
        log_soft_student = F.log_softmax(student_logits / T, dim=1)
        kd = F.kl_div(log_soft_student, soft_targets, reduction='batchmean') * (T * T)
        
        losses['kd'] = kd.item()
        total_loss += self.alpha * kd
        
        # Feature distillation (average teacher features)
        if self.gamma > 0 and teacher_features_list is not None:
            avg_teacher_features = torch.stack(teacher_features_list, dim=0).mean(dim=0)
            feat = self.feat_loss(student_features, avg_teacher_features)
            losses['feat'] = feat.item()
            total_loss += self.gamma * feat
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


# =============================================================================
# AGRE-KD LOSS (Adaptive Group Robust Ensemble KD)
# =============================================================================

class AGREKDLoss(nn.Module):
    """
    AGRE-KD: Adaptive Group Robust Ensemble Knowledge Distillation.

    Uses gradient-based per-sample teacher weighting:
    W_t(x) = 1 - cos_sim(grad_teacher_t, grad_biased)

    Teachers whose gradients align with the biased model get lower weights,
    encouraging the student to learn from teachers that disagree with bias.

    Reference: Kenfack et al., 2024 (arXiv:2411.14984)

    Args:
        alpha: Weight for KD vs CE (1.0 = pure KD, 0.0 = pure CE)
        gamma: Weight for single-layer feature distillation (backward compatible)
        layer_gammas: Dict of {layer_name: gamma} for multi-layer distillation
                      If provided, overrides 'gamma' parameter
        temperature: Temperature for KD loss
        student_dim: Student feature dimension (for single-layer adapter)
        teacher_dim: Teacher feature dimension (for single-layer adapter)
        student_arch: Student architecture for multi-layer ('resnet18')
        teacher_arch: Teacher architecture for multi-layer ('resnet50')
        use_disagreement_weighting: If True, weight feature dimensions by teacher
                                    agreement (high agreement = high weight)
    """

    def __init__(self, alpha=1.0, gamma=0.0, layer_gammas=None, temperature=4.0,
                 student_dim=512, teacher_dim=2048,
                 student_arch='resnet18', teacher_arch='resnet50',
                 use_disagreement_weighting=False):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.use_disagreement_weighting = use_disagreement_weighting

        self.ce_loss = nn.CrossEntropyLoss()

        # Multi-layer feature distillation
        if layer_gammas is not None and any(g > 0 for g in layer_gammas.values()):
            self.feat_loss = MultiLayerFeatureDistillationLoss(
                layer_gammas=layer_gammas,
                student_arch=student_arch,
                teacher_arch=teacher_arch
            )
            self.gamma = self.feat_loss.total_gamma
            self.use_multilayer = True
        # Single-layer feature distillation
        elif gamma > 0:
            # Use disagreement-weighted loss if enabled
            if use_disagreement_weighting:
                self.feat_loss = DisagreementWeightedFeatureLoss(
                    student_dim=student_dim,
                    teacher_dim=teacher_dim
                )
            else:
                self.feat_loss = FeatureDistillationLoss(
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    spatial=False
                )
            self.gamma = gamma
            self.use_multilayer = False
        else:
            self.feat_loss = None
            self.gamma = 0.0
            self.use_multilayer = False

    def compute_kd_loss(self, student_logits, teacher_logits):
        """Compute KD loss between student and single teacher."""
        T = self.temperature
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        log_soft_student = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_soft_student, soft_targets, reduction='batchmean') * (T * T)

    def compute_teacher_weights(self, student, teacher_logits_list, biased_logits,
                                 student_logits):
        """
        Compute per-teacher weights based on gradient alignment with biased model.

        Teachers whose gradients point in the same direction as the biased model
        are downweighted (they're likely reinforcing spurious correlations).

        Args:
            student: Student model (for gradient computation)
            teacher_logits_list: List of teacher logits
            biased_logits: Logits from biased reference model
            student_logits: Current student logits (requires_grad=True)

        Returns:
            weights: [num_teachers] tensor of normalized weights
        """
        num_teachers = len(teacher_logits_list)
        weights = []

        # Compute gradient direction for biased model
        biased_loss = self.compute_kd_loss(student_logits, biased_logits.detach())
        student.zero_grad()
        biased_loss.backward(retain_graph=True)
        biased_grad = self._flatten_grads(student)
        biased_grad = F.normalize(biased_grad, dim=0)

        # Compute weight for each teacher
        for t_logits in teacher_logits_list:
            teacher_loss = self.compute_kd_loss(student_logits, t_logits.detach())
            student.zero_grad()
            teacher_loss.backward(retain_graph=True)
            teacher_grad = self._flatten_grads(student)
            teacher_grad = F.normalize(teacher_grad, dim=0)

            # Weight = 1 - cosine_similarity
            # Higher weight for teachers that disagree with biased model
            cos_sim = torch.dot(teacher_grad, biased_grad)
            weight = 1.0 - cos_sim.item()
            weights.append(max(weight, 0.01))  # Ensure positive

        # Normalize weights to sum to 1
        weights = torch.tensor(weights, device=student_logits.device)
        weights = weights / weights.sum()

        student.zero_grad()  # Clean up
        return weights

    def _flatten_grads(self, model):
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if grads:
            return torch.cat(grads)
        else:
            return torch.zeros(1, device=next(model.parameters()).device)

    def forward(self, student_logits, teacher_logits_list, labels,
                student_features=None, teacher_features_list=None,
                teacher_weights=None):
        """
        Compute AGRE-KD loss with pre-computed or uniform weights.

        For efficiency, compute weights separately using compute_teacher_weights()
        before calling forward(), then pass them here. If weights=None, uses
        uniform weighting (equivalent to AVER baseline).

        Args:
            student_logits: Student predictions [B, C]
            teacher_logits_list: List of teacher predictions, each [B, C]
            labels: Ground truth labels [B]
            student_features: Student features [B, D] (optional, for gamma > 0)
            teacher_features_list: List of teacher features, each [B, D]
            teacher_weights: Pre-computed weights [num_teachers] (optional)

        Returns:
            loss: Combined loss
            loss_dict: Dict with component losses for logging
        """
        losses = {}
        total_loss = 0.0
        T = self.temperature
        num_teachers = len(teacher_logits_list)

        # Cross-entropy with ground truth labels
        if self.alpha < 1.0:
            ce = self.ce_loss(student_logits, labels)
            losses['ce'] = ce.item()
            total_loss += (1 - self.alpha) * ce

        # Use uniform weights if not provided (AVER baseline)
        if teacher_weights is None:
            teacher_weights = torch.ones(num_teachers, device=student_logits.device)
            teacher_weights = teacher_weights / num_teachers

        # Weighted ensemble KD loss
        kd_loss = 0.0
        for i, t_logits in enumerate(teacher_logits_list):
            t_loss = self.compute_kd_loss(student_logits, t_logits.detach())
            kd_loss += teacher_weights[i] * t_loss

        losses['kd'] = kd_loss.item()
        total_loss += self.alpha * kd_loss

        # Feature distillation (weighted average of teacher features)
        if self.gamma > 0 and self.feat_loss is not None and teacher_features_list is not None:
            if self.use_multilayer:
                # Multi-layer: features are dicts {layer_name: tensor}
                # Average teacher features per layer
                avg_teacher_features = {}
                layers = self.feat_loss.gammas.keys()

                for layer in layers:
                    # Stack this layer's features from all teachers
                    layer_feats = [t_feat[layer] for t_feat in teacher_features_list]
                    stacked = torch.stack(layer_feats, dim=0)

                    # Weighted average
                    if stacked.dim() == 5:  # [T, B, C, H, W] for spatial
                        w = teacher_weights.view(-1, 1, 1, 1, 1)
                    else:  # [T, B, D] for pooled
                        w = teacher_weights.view(-1, 1, 1)
                    avg_teacher_features[layer] = (stacked * w).sum(dim=0)

                # Compute multi-layer loss
                feat_loss, feat_losses = self.feat_loss(student_features, avg_teacher_features)
                losses.update(feat_losses)
                total_loss += feat_loss

            elif self.use_disagreement_weighting:
                # Disagreement weighting: pass full list for variance computation
                feat = self.feat_loss(student_features, teacher_features_list)
                losses['feat'] = feat.item()
                total_loss += self.gamma * feat

            else:
                # Standard single-layer: weighted average of teacher features
                # Stack teacher features: [num_teachers, B, D]
                stacked = torch.stack(teacher_features_list, dim=0)

                # Weighted average: [B, D]
                w = teacher_weights.view(-1, 1, 1)
                avg_teacher_feat = (stacked * w).sum(dim=0)

                feat = self.feat_loss(student_features, avg_teacher_feat)
                losses['feat'] = feat.item()
                total_loss += self.gamma * feat

        losses['total'] = total_loss.item()
        losses['weights'] = teacher_weights.cpu().tolist()  # For logging

        return total_loss, losses


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing loss functions...")
    
    B, C = 8, 2  # batch size, num classes
    D_s, D_t = 512, 2048  # student/teacher dims
    
    # Fake data
    student_logits = torch.randn(B, C)
    teacher_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))
    student_features = torch.randn(B, D_s)
    teacher_features = torch.randn(B, D_t)
    
    # Test Experiment 1: alpha=0.7, gamma=0 (class labels)
    print("\nExperiment 1 (alpha=0.7, gamma=0):")
    loss_fn = CombinedDistillationLoss(alpha=0.7, gamma=0.0)
    loss, loss_dict = loss_fn(student_logits, teacher_logits, labels)
    print(f"  Losses: {loss_dict}")
    
    # Test Experiment 2: alpha=1, gamma=0.1 (feature distillation)
    print("\nExperiment 2 (alpha=1.0, gamma=0.1):")
    loss_fn = CombinedDistillationLoss(alpha=1.0, gamma=0.1,
                                        student_dim=D_s, teacher_dim=D_t)
    loss, loss_dict = loss_fn(student_logits, teacher_logits, labels,
                               student_features, teacher_features)
    print(f"  Losses: {loss_dict}")
    
    # Test Experiment 3: alpha=0.7, gamma=0.1 (combined)
    print("\nExperiment 3 (alpha=0.7, gamma=0.1):")
    loss_fn = CombinedDistillationLoss(alpha=0.7, gamma=0.1,
                                        student_dim=D_s, teacher_dim=D_t)
    loss, loss_dict = loss_fn(student_logits, teacher_logits, labels,
                               student_features, teacher_features)
    print(f"  Losses: {loss_dict}")
    
    # Test Ensemble KD
    print("\nEnsemble KD (3 teachers):")
    teacher_logits_list = [torch.randn(B, C) for _ in range(3)]
    teacher_features_list = [torch.randn(B, D_t) for _ in range(3)]
    
    ensemble_loss = EnsembleKDLoss(alpha=0.7, gamma=0.1,
                                   student_dim=D_s, teacher_dim=D_t)
    loss, loss_dict = ensemble_loss(student_logits, teacher_logits_list, labels,
                                     student_features, teacher_features_list)
    print(f"  Losses: {loss_dict}")
