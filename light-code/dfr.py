"""
Deep Feature Reweighting (DFR) Implementation
Based on Kirichenko et al. (2023): "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations"

DFR is a simple but effective debiasing technique:
1. Train model with ERM (learns spurious correlations)
2. Freeze feature extractor
3. Retrain ONLY the last layer on group-balanced data

This transforms biased ERM teachers (~70% WGA) into debiased teachers (~92% WGA).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(model, dataloader, device='cuda', verbose=True):
    """
    Extract penultimate layer features from a model.

    Args:
        model: Model with forward(x, return_features=True) method
        dataloader: DataLoader yielding {'image', 'label', 'group'} dicts
        device: Device to run on
        verbose: If True, show progress bar

    Returns:
        features: np.array [N, D] of penultimate features
        labels: np.array [N] of class labels
        groups: np.array [N] of group labels
    """
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []
    all_groups = []

    iterator = tqdm(dataloader, desc="        Batches", disable=not verbose, leave=False)

    with torch.no_grad():
        for batch in iterator:
            images = batch['image'].to(device)
            labels = batch['label']
            groups = batch['group']
            
            # Get penultimate features
            _, features = model(images, return_features=True)
            pooled = features['pooled'].cpu().numpy()
            
            all_features.append(pooled)
            all_labels.append(labels.numpy())
            all_groups.append(groups.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    return features, labels, groups


# =============================================================================
# BALANCED SUBSET CREATION
# =============================================================================

def create_balanced_subset(features, labels, groups, balance_type='group'):
    """
    Create a balanced subset for DFR training.

    Args:
        features: [N, D] feature array
        labels: [N] label array
        groups: [N] group array
        balance_type: 'group' (balance all 4 groups) or 'class' (balance 2 classes)

    Returns:
        balanced_features: Balanced subset of features
        balanced_labels: Balanced subset of labels
    """
    if balance_type == 'group':
        # Balance all 4 groups (most aggressive debiasing)
        unique_groups = np.unique(groups)
        min_count = min(np.sum(groups == g) for g in unique_groups)

        indices = []
        for g in unique_groups:
            group_indices = np.where(groups == g)[0]
            # Subsample to min_count
            selected = np.random.choice(group_indices, size=min_count, replace=False)
            indices.extend(selected)

        indices = np.array(indices)

    elif balance_type == 'class':
        # Balance 2 classes only
        unique_labels = np.unique(labels)
        min_count = min(np.sum(labels == l) for l in unique_labels)

        indices = []
        for l in unique_labels:
            label_indices = np.where(labels == l)[0]
            selected = np.random.choice(label_indices, size=min_count, replace=False)
            indices.extend(selected)

        indices = np.array(indices)

    else:
        # No balancing
        indices = np.arange(len(labels))

    np.random.shuffle(indices)
    return features[indices], labels[indices]


# =============================================================================
# DFR TRAINING (SKLEARN - FAST)
# =============================================================================

def train_dfr_sklearn(features, labels, C=1.0, class_weight='balanced'):
    """
    Train DFR classifier using sklearn LogisticRegression.

    This is FAST (~10 seconds) and works well.

    Args:
        features: [N, D] balanced feature array
        labels: [N] balanced label array
        C: Regularization parameter (higher = less regularization)
        class_weight: 'balanced' or None

    Returns:
        clf: Trained LogisticRegression classifier
    """
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )

    clf.fit(features, labels)

    return clf


# =============================================================================
# DFR TRAINING (PYTORCH - MORE CONTROL)
# =============================================================================

def train_dfr_pytorch(features, labels, feature_dim, num_classes=2,
                      lr=0.01, epochs=100, batch_size=256, device='cuda'):
    """
    Train DFR classifier using PyTorch.
    
    Gives more control over training but slower than sklearn.
    
    Args:
        features: [N, D] balanced feature array
        labels: [N] balanced label array
        feature_dim: Dimension of features (e.g., 2048 for ResNet-50)
        num_classes: Number of classes
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        classifier: Trained nn.Linear module
    """
    print(f"Training DFR classifier (PyTorch, lr={lr}, epochs={epochs})...")
    
    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Simple linear classifier
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={epoch_loss/len(loader):.4f}")
    
    # Check training accuracy
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(X.to(device))
        preds = outputs.argmax(dim=1).cpu()
        train_acc = (preds == y).float().mean()
        print(f"  Training accuracy: {train_acc*100:.2f}%")
    
    return classifier


# =============================================================================
# APPLY DFR TO MODEL
# =============================================================================

def apply_dfr(model, val_loader, device='cuda', method='sklearn',
              balance_type='group', C=1.0, verbose=True):
    """
    Apply DFR to debias a model by retraining its last layer.

    This is the main function you'll use.

    Args:
        model: Trained ERM model (biased)
        val_loader: Validation dataloader (used for balanced training)
        device: Device
        method: 'sklearn' (fast) or 'pytorch' (more control)
        balance_type: 'group' or 'class'
        C: Regularization for sklearn method
        verbose: If True, show progress

    Returns:
        model: Same model with debiased last layer (modified in-place)
    """
    import time

    total_samples = len(val_loader.dataset)

    # Step 1: Extract features
    if verbose:
        print(f"      [1/3] Extracting features from {total_samples} samples...")
    t0 = time.time()
    features, labels, groups = extract_features(model, val_loader, device, verbose=verbose)
    if verbose:
        print(f"            Features: {features.shape} ({time.time()-t0:.1f}s)")

    # Step 2: Create balanced subset
    if verbose:
        print("      [2/3] Creating balanced subset...")
    t0 = time.time()
    balanced_features, balanced_labels = create_balanced_subset(
        features, labels, groups, balance_type
    )
    if verbose:
        print(f"            {len(balanced_labels)} samples ({time.time()-t0:.1f}s)")

    # Step 3: Train new classifier
    if verbose:
        print("      [3/3] Training new classifier...")
    t0 = time.time()

    if method == 'sklearn':
        clf = train_dfr_sklearn(balanced_features, balanced_labels, C=C)

        # Replace model's last layer
        with torch.no_grad():
            # Handle binary classification: sklearn returns [1, D] for 2 classes
            # Convert to PyTorch [2, D] format
            if clf.coef_.shape[0] == 1:
                # Binary classification: sklearn gives [1, D], need [2, D]
                # Class 0 logit = 0, Class 1 logit = sklearn decision function
                new_weight = torch.zeros(2, clf.coef_.shape[1], dtype=torch.float32)
                new_bias = torch.zeros(2, dtype=torch.float32)
                new_weight[1] = torch.tensor(clf.coef_[0], dtype=torch.float32)
                new_bias[1] = torch.tensor(clf.intercept_[0], dtype=torch.float32)
            else:
                # Multi-class: sklearn gives [C, D], use directly
                new_weight = torch.tensor(clf.coef_, dtype=torch.float32)
                new_bias = torch.tensor(clf.intercept_, dtype=torch.float32)

            # Copy to model (handles device transfer)
            model.fc.weight.copy_(new_weight.to(model.fc.weight.device))
            model.fc.bias.copy_(new_bias.to(model.fc.bias.device))

    elif method == 'pytorch':
        feature_dim = features.shape[1]
        num_classes = len(np.unique(labels))

        classifier = train_dfr_pytorch(
            balanced_features, balanced_labels,
            feature_dim, num_classes,
            device=device
        )

        # Replace model's last layer
        with torch.no_grad():
            model.fc.weight.copy_(classifier.weight.data)
            model.fc.bias.copy_(classifier.bias.data)

    if verbose:
        print(f"            Done ({time.time()-t0:.1f}s)")

    return model


# =============================================================================
# BATCH DFR FOR MULTIPLE TEACHERS
# =============================================================================

def apply_dfr_to_teachers(teachers, val_loader, device='cuda', method='sklearn',
                          balance_type='group', C=1.0):
    """
    Apply DFR to multiple teacher models.
    
    Since all teachers have the same architecture and were trained on the same
    data, we can extract features once and train separate classifiers.
    
    Args:
        teachers: List of teacher models
        val_loader: Validation dataloader
        device: Device
        method: 'sklearn' or 'pytorch'
        balance_type: 'group' or 'class'
        C: Regularization for sklearn
        
    Returns:
        debiased_teachers: List of debiased teacher models
    """
    print(f"\nApplying DFR to {len(teachers)} teachers...")
    
    debiased_teachers = []
    
    for i, teacher in enumerate(teachers):
        print(f"\n--- Teacher {i+1}/{len(teachers)} ---")
        
        # Apply DFR (modifies in-place, but we'll keep a reference)
        apply_dfr(
            teacher, val_loader, device=device,
            method=method, balance_type=balance_type, C=C
        )
        
        debiased_teachers.append(teacher)
    
    print(f"\nDFR applied to all {len(teachers)} teachers!")
    
    return debiased_teachers


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    import argparse
    from data import get_waterbirds_loaders
    from models import get_teacher_model, load_teacher_checkpoint
    from eval import compute_group_accuracies, print_results
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to ERM teacher checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Waterbirds data')
    parser.add_argument('--output', type=str, default='./teacher_debiased.pt',
                        help='Output path for debiased model')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("Loading data...")
    loaders = get_waterbirds_loaders(args.data_dir, batch_size=64)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_teacher_model('resnet50', num_classes=2, pretrained=False)
    load_teacher_checkpoint(model, args.checkpoint)
    model = model.to(device)
    
    # Evaluate before DFR
    print("\nBefore DFR:")
    results_before = compute_group_accuracies(model, loaders['test'], device)
    print_results(results_before, "Before DFR")
    
    # Apply DFR
    apply_dfr(model, loaders['val'], device=device, method='sklearn', balance_type='group')
    
    # Evaluate after DFR
    print("\nAfter DFR:")
    results_after = compute_group_accuracies(model, loaders['test'], device)
    print_results(results_after, "After DFR")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'wga_before': results_before['wga'],
        'wga_after': results_after['wga'],
    }, args.output)
    print(f"\nSaved debiased model to {args.output}")
