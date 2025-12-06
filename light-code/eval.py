"""
Evaluation metrics for group robustness experiments.
Computes per-group accuracy, worst-group accuracy (WGA), and average accuracy.
"""

import time
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def compute_group_accuracies(model, dataloader, device='cuda', verbose=True):
    """
    Compute per-group and aggregate accuracy metrics.

    Args:
        model: Trained model
        dataloader: DataLoader with 'image', 'label', 'group' keys
        device: Device to run evaluation on
        verbose: If True, show progress bar and timing

    Returns:
        results: Dict with per-group accuracy, WGA, and average accuracy
    """
    model.eval()

    total_samples = len(dataloader.dataset)
    total_batches = len(dataloader)

    if verbose:
        print(f"  Evaluating {total_samples} samples ({total_batches} batches)...")

    start_time = time.time()

    # Track predictions and labels by group
    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    all_preds = []
    all_labels = []
    all_groups = []

    iterator = tqdm(dataloader, desc="  Batches", disable=not verbose, leave=False)

    with torch.no_grad():
        for batch in iterator:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            groups = batch['group'].to(device)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Handle (logits, features) return
            
            preds = outputs.argmax(dim=1)
            
            # Track per-group stats
            for i in range(len(labels)):
                g = groups[i].item()
                correct = (preds[i] == labels[i]).item()
                group_correct[g] += correct
                group_total[g] += 1
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_groups.append(groups.cpu())
    
    # Compute per-group accuracy
    group_accs = {}
    for g in sorted(group_total.keys()):
        group_accs[g] = group_correct[g] / group_total[g] if group_total[g] > 0 else 0.0
    
    # Worst-group accuracy (the key metric!)
    wga = min(group_accs.values())
    worst_group = min(group_accs, key=group_accs.get)
    
    # Average accuracy (overall)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_acc = (all_preds == all_labels).float().mean().item()
    
    # Accuracy gap
    acc_gap = max(group_accs.values()) - min(group_accs.values())

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Evaluation complete ({elapsed:.1f}s) - WGA: {wga*100:.1f}%")

    return {
        'group_accs': group_accs,
        'wga': wga,
        'worst_group': worst_group,
        'avg_acc': avg_acc,
        'acc_gap': acc_gap,
        'group_counts': dict(group_total),
    }


def print_results(results, title="Evaluation Results"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    print(f"\nPer-group accuracy:")
    group_names = {
        0: "Landbird + Land (majority)",
        1: "Landbird + Water (minority)",
        2: "Waterbird + Land (minority, hardest)",
        3: "Waterbird + Water (majority)",
    }
    for g, acc in results['group_accs'].items():
        name = group_names.get(g, f"Group {g}")
        count = results['group_counts'].get(g, '?')
        print(f"  {name}: {acc*100:.2f}% (n={count})")
    
    print(f"\nAggregate metrics:")
    print(f"  Worst-Group Accuracy (WGA): {results['wga']*100:.2f}%")
    print(f"  Average Accuracy: {results['avg_acc']*100:.2f}%")
    print(f"  Accuracy Gap: {results['acc_gap']*100:.2f}%")
    print(f"  Worst Group: {results['worst_group']}")
    print(f"{'='*50}\n")


def evaluate_ensemble(models, dataloader, device='cuda'):
    """
    Evaluate an ensemble of models by averaging predictions.
    
    Args:
        models: List of trained models
        dataloader: DataLoader
        device: Device
        
    Returns:
        results: Same as compute_group_accuracies
    """
    for model in models:
        model.eval()
    
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            groups = batch['group'].to(device)
            
            # Average predictions across ensemble
            all_logits = []
            for model in models:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                all_logits.append(outputs)
            
            avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            preds = avg_logits.argmax(dim=1)
            
            for i in range(len(labels)):
                g = groups[i].item()
                correct = (preds[i] == labels[i]).item()
                group_correct[g] += correct
                group_total[g] += 1
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Same aggregation as single model
    group_accs = {g: group_correct[g] / group_total[g] for g in sorted(group_total.keys())}
    wga = min(group_accs.values())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_acc = (all_preds == all_labels).float().mean().item()
    
    return {
        'group_accs': group_accs,
        'wga': wga,
        'worst_group': min(group_accs, key=group_accs.get),
        'avg_acc': avg_acc,
        'acc_gap': max(group_accs.values()) - min(group_accs.values()),
        'group_counts': dict(group_total),
    }


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class MetricLogger:
    """Simple logger for tracking metrics during training."""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def log(self, metrics, step=None):
        """Log metrics dict."""
        for k, v in metrics.items():
            self.history[k].append(v)
    
    def get_last(self, key):
        """Get most recent value for a metric."""
        return self.history[key][-1] if self.history[key] else None
    
    def get_best(self, key, mode='max'):
        """Get best value for a metric."""
        if not self.history[key]:
            return None
        return max(self.history[key]) if mode == 'max' else min(self.history[key])
    
    def to_dict(self):
        """Export as dict."""
        return dict(self.history)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing evaluation metrics...")
    
    # Create fake results
    fake_results = {
        'group_accs': {0: 0.95, 1: 0.82, 2: 0.68, 3: 0.94},
        'wga': 0.68,
        'worst_group': 2,
        'avg_acc': 0.93,
        'acc_gap': 0.27,
        'group_counts': {0: 2255, 1: 642, 2: 133, 3: 466},
    }
    
    print_results(fake_results, "Example Results (Fake)")
