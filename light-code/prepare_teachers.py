"""
Prepare Teacher Models for AGRE-KD Experiments

This script:
1. Loads ERM checkpoints (erm_seed{N}.pt) from teacher_checkpoints folder
2. Applies DFR to create debiased versions (teacher_{N}_debiased.pt)
3. Saves debiased teachers to the SAME folder

Folder structure after running:
    teacher_checkpoints/
    ├── erm_seed0.pt              # Biased (original)
    ├── erm_seed1.pt              # Biased (original)
    ├── teacher_0_debiased.pt     # Debiased (created by this script)
    ├── teacher_1_debiased.pt     # Debiased (created by this script)
    └── ...

DFR Checkpoints Source:
https://drive.google.com/drive/folders/1OQ_oPPgxgK_7j_GCt71znyiRj6hqi_UW

Usage:
    python prepare_teachers.py \\
        --checkpoint_dir ./teacher_checkpoints \\
        --data_dir ./data/waterbirds_v1.0
"""

import os
import re
import argparse
import torch

from data import get_waterbirds_loaders
from models import get_teacher_model, load_dfr_checkpoint
from eval import compute_group_accuracies
from dfr import apply_dfr


def find_erm_checkpoints(checkpoint_dir):
    """Find all ERM checkpoint files and extract their seed numbers."""
    checkpoints = []
    pattern = re.compile(r'erm_seed(\d+)\.pt')
    
    for f in sorted(os.listdir(checkpoint_dir)):
        match = pattern.match(f)
        if match:
            seed_num = int(match.group(1))
            checkpoints.append({
                'path': os.path.join(checkpoint_dir, f),
                'seed': seed_num,
                'filename': f
            })
    
    return checkpoints


def prepare_single_teacher(erm_info, loaders, device='cuda'):
    """
    Apply DFR to create a debiased version of an ERM teacher.

    Args:
        erm_info: Dict with 'path', 'seed', 'filename' for ERM checkpoint
        loaders: Data loaders dict
        device: Device to use

    Returns:
        results: Dict with evaluation results
    """
    import time

    checkpoint_path = erm_info['path']
    seed_num = erm_info['seed']
    checkpoint_dir = os.path.dirname(checkpoint_path)
    step_times = {}
    total_start = time.time()

    # Step 1: Load model
    print("  [1/5] Loading model...")
    t0 = time.time()
    model = get_teacher_model('resnet50', num_classes=2, pretrained=False)
    load_dfr_checkpoint(model, checkpoint_path)
    model = model.to(device)
    step_times['load'] = time.time() - t0
    print(f"        Done ({step_times['load']:.1f}s)")

    # Step 2: Evaluate biased version
    print("  [2/5] Evaluating biased (ERM) model...")
    t0 = time.time()
    model.eval()
    biased_results = compute_group_accuracies(model, loaders['test'], device, verbose=False)
    step_times['eval_biased'] = time.time() - t0
    print(f"        WGA: {biased_results['wga']*100:.1f}% ({step_times['eval_biased']:.1f}s)")

    # Step 3: Apply DFR
    print("  [3/5] Applying DFR (retraining last layer)...")
    t0 = time.time()
    apply_dfr(model, loaders['val'], device=device, method='sklearn', balance_type='group', verbose=True)
    step_times['dfr'] = time.time() - t0
    print(f"        Done ({step_times['dfr']:.1f}s)")

    # Step 4: Evaluate debiased version
    print("  [4/5] Evaluating debiased (DFR) model...")
    t0 = time.time()
    model.eval()
    debiased_results = compute_group_accuracies(model, loaders['test'], device, verbose=False)
    step_times['eval_debiased'] = time.time() - t0
    print(f"        WGA: {debiased_results['wga']*100:.1f}% ({step_times['eval_debiased']:.1f}s)")

    # Step 5: Save checkpoint
    print("  [5/5] Saving checkpoint...")
    t0 = time.time()
    debiased_filename = f'teacher_{seed_num}_debiased.pt'
    debiased_path = os.path.join(checkpoint_dir, debiased_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'wga': debiased_results['wga'],
        'avg_acc': debiased_results['avg_acc'],
        'group_accs': debiased_results['group_accs'],
        'type': 'debiased',
        'source': checkpoint_path,
        'seed': seed_num,
        'biased_wga': biased_results['wga'],
    }, debiased_path)
    step_times['save'] = time.time() - t0
    print(f"        Saved: {debiased_filename} ({step_times['save']:.1f}s)")

    # Summary
    total_time = time.time() - total_start
    improvement = (debiased_results['wga'] - biased_results['wga']) * 100
    print(f"\n  Summary: {biased_results['wga']*100:.1f}% -> {debiased_results['wga']*100:.1f}% "
          f"(+{improvement:.1f}%) in {total_time:.1f}s")

    return {
        'seed': seed_num,
        'biased': biased_results,
        'debiased': debiased_results,
        'biased_path': checkpoint_path,
        'debiased_path': debiased_path,
    }


def prepare_all_teachers(checkpoint_dir, data_dir, num_teachers=None, device='cuda'):
    """
    Prepare all teachers from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing erm_seed{N}.pt files
        data_dir: Path to Waterbirds data
        num_teachers: Max number of teachers (None = all)
        device: Device to use

    Returns:
        all_results: Dict mapping seed to results
    """
    import time

    total_start = time.time()

    # Find ERM checkpoints
    erm_checkpoints = find_erm_checkpoints(checkpoint_dir)

    if len(erm_checkpoints) == 0:
        print(f"ERROR: No erm_seed*.pt files found in {checkpoint_dir}")
        return None

    if num_teachers is not None:
        erm_checkpoints = erm_checkpoints[:num_teachers]

    total = len(erm_checkpoints)
    print(f"\n{'='*60}")
    print(f"PREPARING {total} TEACHERS")
    print(f"{'='*60}")
    print(f"\nFound {total} ERM checkpoints:")
    for info in erm_checkpoints:
        print(f"  - {info['filename']} (seed {info['seed']})")

    # Load data (once for all teachers)
    print("\nLoading data...")
    data_start = time.time()
    loaders = get_waterbirds_loaders(data_dir, batch_size=64, num_workers=2)
    print(f"Data loaded ({time.time() - data_start:.1f}s)")

    # Process each checkpoint
    all_results = {}

    for i, erm_info in enumerate(erm_checkpoints):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Processing {erm_info['filename']}")
        print(f"{'='*60}")

        teacher_start = time.time()
        results = prepare_single_teacher(erm_info, loaders, device)
        all_results[erm_info['seed']] = results

        teacher_time = time.time() - teacher_start
        elapsed_total = time.time() - total_start
        remaining = (elapsed_total / (i + 1)) * (total - i - 1)

        print(f"\n[{i+1}/{total}] Done in {teacher_time:.1f}s "
              f"(elapsed: {elapsed_total/60:.1f}min, remaining: ~{remaining/60:.1f}min)")

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TEACHER PREPARATION COMPLETE ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\nAll teachers in: {checkpoint_dir}")
    print("\nBiased (ERM):")
    for seed, res in sorted(all_results.items()):
        print(f"  erm_seed{seed}.pt -> WGA={res['biased']['wga']*100:.1f}%")

    print("\nDebiased (DFR):")
    for seed, res in sorted(all_results.items()):
        print(f"  teacher_{seed}_debiased.pt -> WGA={res['debiased']['wga']*100:.1f}%")

    avg_improvement = sum(
        (r['debiased']['wga'] - r['biased']['wga']) * 100
        for r in all_results.values()
    ) / len(all_results)
    print(f"\nAverage WGA improvement: +{avg_improvement:.1f}%")

    # Save summary
    summary_path = os.path.join(checkpoint_dir, 'preparation_summary.pt')
    torch.save({
        'num_teachers': len(all_results),
        'results': all_results,
        'total_time': total_time,
    }, summary_path)
    print(f"Summary saved to {summary_path}")

    return all_results


# =============================================================================
# COLAB HELPERS
# =============================================================================

def download_dfr_checkpoints_instructions():
    """Print instructions for downloading DFR checkpoints."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DOWNLOADING DFR CHECKPOINTS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  The DFR authors provide pre-trained ERM checkpoints on Google Drive:        ║
║                                                                              ║
║  https://drive.google.com/drive/folders/1OQ_oPPgxgK_7j_GCt71znyiRj6hqi_UW   ║
║                                                                              ║
║  Steps for Google Colab:                                                     ║
║  1. Open the link above                                                      ║
║  2. Navigate to: spurious_feature_learning/results/waterbirds_paper          ║
║  3. Download the checkpoint files (erm_seed*.pt)                             ║
║  4. Upload to your Google Drive: MyDrive/robust-ensemble-kd/teacher_checkpoints/
║                                                                              ║
║  Or use gdown in Colab:                                                      ║
║  !pip install gdown                                                          ║
║  !gdown --folder <folder_id> -O /content/drive/MyDrive/.../teacher_checkpoints
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def colab_prepare_teachers(checkpoint_dir, data_dir, num_teachers=5):
    """
    Convenience function for Google Colab.
    
    Example usage in Colab:
        from prepare_teachers import colab_prepare_teachers
        results = colab_prepare_teachers(
            checkpoint_dir='/content/drive/MyDrive/robust-ensemble-kd/teacher_checkpoints',
            data_dir='/content/drive/MyDrive/robust-ensemble-kd/data/waterbirds_v1.0',
            num_teachers=5
        )
    """
    # Check paths
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        download_dfr_checkpoints_instructions()
        return None
    
    # Check for ERM files
    erm_files = find_erm_checkpoints(checkpoint_dir)
    if len(erm_files) == 0:
        print(f"ERROR: No erm_seed*.pt files found in {checkpoint_dir}")
        download_dfr_checkpoints_instructions()
        return None
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Download Waterbirds with: from data import download_waterbirds; download_waterbirds('./data')")
        return None
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU - this will be slow!")
    
    # Run preparation
    return prepare_all_teachers(
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        num_teachers=num_teachers,
        device=device
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare teacher models for AGRE-KD')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing erm_seed*.pt files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Waterbirds data')
    parser.add_argument('--num_teachers', type=int, default=None,
                        help='Number of teachers to prepare (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    prepare_all_teachers(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        num_teachers=args.num_teachers,
        device=args.device
    )
