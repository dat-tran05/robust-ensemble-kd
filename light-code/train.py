"""
Training loops for teachers and students.
Handles checkpointing, resuming, and logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_waterbirds_loaders
from models import get_teacher_model, get_student_model, create_feature_adapter, load_checkpoint, load_teacher_checkpoint
from losses import CombinedDistillationLoss, EnsembleKDLoss, AGREKDLoss
from eval import compute_group_accuracies, print_results, MetricLogger
from config import Config


# =============================================================================
# TEACHER TRAINING (ERM)
# =============================================================================

def train_teacher(config, checkpoint_path=None):
    """
    Train a single teacher model with ERM (standard cross-entropy).
    
    Args:
        config: Config object with hyperparameters
        checkpoint_path: Optional path to resume from
        
    Returns:
        model: Trained model
        history: Training history
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Training teacher on {device}")
    
    # Data
    loaders = get_waterbirds_loaders(
        config.data_dir, 
        batch_size=config.batch_size,
        augment=config.augment
    )
    
    # Model
    model = get_teacher_model(
        config.teacher_arch, 
        config.num_classes, 
        config.pretrained
    ).to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint
    start_epoch = 0
    best_wga = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_wga = ckpt.get('best_wga', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger = MetricLogger()
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Evaluate
        val_results = compute_group_accuracies(model, loaders['val'], device)
        
        # Log
        logger.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(loaders['train']),
            'val_wga': val_results['wga'],
            'val_avg': val_results['avg_acc'],
        })
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(loaders['train']):.4f}, "
              f"Val WGA={val_results['wga']*100:.2f}%, "
              f"Val Avg={val_results['avg_acc']*100:.2f}%")
        
        # Save best
        if val_results['wga'] > best_wga:
            best_wga = val_results['wga']
            save_path = os.path.join(config.checkpoint_dir, f'teacher_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wga': best_wga,
                'config': config,
            }, save_path)
            print(f"  New best WGA: {best_wga*100:.2f}%")
        
        # Periodic checkpoint (for resuming)
        if (epoch + 1) % config.save_freq == 0:
            save_path = os.path.join(config.checkpoint_dir, f'teacher_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wga': best_wga,
            }, save_path)
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_results = compute_group_accuracies(model, loaders['test'], device)
    print_results(test_results, "Teacher Test Results")
    
    return model, logger.to_dict()


# =============================================================================
# STUDENT DISTILLATION
# =============================================================================

def train_student(config, teachers, biased_model=None, exp_name='baseline',
                  use_agre=True, checkpoint_path=None):
    """
    Train a student model via knowledge distillation from teacher ensemble.

    Supports both AVER (simple averaging) and AGRE-KD (gradient-based weighting).

    Args:
        config: Config object (with alpha, gamma, etc.)
        teachers: List of teacher models (already loaded, in eval mode)
        biased_model: Reference biased model for AGRE-KD (if None, uses teachers[0])
        exp_name: Experiment name for saving
        use_agre: If True, use AGRE-KD gradient weighting; else use simple averaging
        checkpoint_path: Optional path to resume from

    Returns:
        student: Trained student model
        history: Training history dict
        test_results: Test evaluation results
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    method = "AGRE-KD" if use_agre else "AVER"
    print(f"\nTraining student: {exp_name} (α={config.alpha}, γ={config.gamma}, method={method})")

    # Use first teacher as biased model if not specified
    if biased_model is None and use_agre:
        biased_model = teachers[0]
        print(f"  Using teachers[0] as biased reference model")

    # Data
    loaders = get_waterbirds_loaders(
        config.data_dir,
        batch_size=config.batch_size,
        augment=config.augment
    )

    # Student model
    student = get_student_model(
        config.student_arch,
        config.num_classes,
        config.pretrained
    ).to(device)

    # Feature adapter (if using feature distillation)
    adapter = None
    if config.gamma > 0:
        adapter = create_feature_adapter(
            config.student_arch,
            config.teacher_arch,
            'pooled'
        ).to(device)

    # Put teachers in eval mode on device
    for t in teachers:
        t.to(device)
        t.eval()

    if biased_model is not None:
        biased_model.to(device)
        biased_model.eval()

    # Loss function - use AGREKDLoss (works for both AGRE and AVER via weights)
    loss_fn = AGREKDLoss(
        alpha=config.alpha,
        gamma=config.gamma,
        temperature=config.temperature,
        student_dim=512 if config.student_arch == 'resnet18' else 2048,
        teacher_dim=2048 if config.teacher_arch == 'resnet50' else 512,
    )

    # Optimizer (include adapter params if present)
    params = list(student.parameters())
    if adapter is not None:
        params += list(adapter.parameters())

    optimizer = optim.SGD(
        params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Resume from checkpoint
    start_epoch = 0
    best_wga = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        student.load_state_dict(ckpt['model_state_dict'])
        if adapter and ckpt.get('adapter_state_dict'):
            adapter.load_state_dict(ckpt['adapter_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_wga = ckpt.get('best_wga', 0.0)
        print(f"  Resumed from epoch {start_epoch}, best WGA: {best_wga*100:.2f}%")

    # Training loop
    logger = MetricLogger()

    for epoch in range(start_epoch, config.epochs):
        student.train()
        if adapter:
            adapter.train()

        epoch_losses = {'total': 0, 'kd': 0, 'ce': 0, 'feat': 0}
        epoch_weights = []

        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Student forward (need gradients for AGRE-KD weight computation)
            s_logits, s_features = student(images, return_features=True)
            s_feat = s_features['pooled']

            # Adapt student features if needed
            if adapter:
                s_feat = adapter(s_feat)

            # Teacher forward (no grad for logits, but need graph for AGRE-KD)
            t_logits_list = []
            t_feat_list = []
            with torch.no_grad():
                for teacher in teachers:
                    t_logits, t_features = teacher(images, return_features=True)
                    t_logits_list.append(t_logits)
                    t_feat_list.append(t_features['pooled'])

            # Compute teacher weights (AGRE-KD or uniform)
            if use_agre and biased_model is not None:
                # Get biased model logits
                with torch.no_grad():
                    biased_logits = biased_model(images)

                # Compute gradient-based weights
                teacher_weights = loss_fn.compute_teacher_weights(
                    student, t_logits_list, biased_logits, s_logits
                )
                epoch_weights.append(teacher_weights.cpu().tolist())
            else:
                teacher_weights = None  # Will use uniform weights

            # Compute loss
            loss, loss_dict = loss_fn(
                s_logits, t_logits_list, labels,
                s_feat if config.gamma > 0 else None,
                t_feat_list if config.gamma > 0 else None,
                teacher_weights=teacher_weights
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            for k, v in loss_dict.items():
                if k != 'weights':  # Don't accumulate weights
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Normalize losses
        n_batches = len(loaders['train'])
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Average weights for logging
        if epoch_weights:
            import numpy as np
            avg_weights = np.mean(epoch_weights, axis=0).tolist()
        else:
            avg_weights = [1.0 / len(teachers)] * len(teachers)

        # Evaluate
        val_results = compute_group_accuracies(student, loaders['val'], device)

        # Log
        logger.log({
            'epoch': epoch + 1,
            **{f'loss_{k}': v for k, v in epoch_losses.items()},
            'val_wga': val_results['wga'],
            'val_avg': val_results['avg_acc'],
            'teacher_weights': avg_weights,
        })

        print(f"Epoch {epoch+1}: Loss={epoch_losses['total']:.4f}, "
              f"Val WGA={val_results['wga']*100:.2f}%, "
              f"Val Avg={val_results['avg_acc']*100:.2f}%")
        if use_agre:
            print(f"  Teacher weights: {[f'{w:.3f}' for w in avg_weights]}")

        # Save best
        if val_results['wga'] > best_wga:
            best_wga = val_results['wga']
            save_path = os.path.join(config.checkpoint_dir, f'student_{exp_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wga': best_wga,
                'val_results': val_results,
                'config': vars(config) if hasattr(config, '__dict__') else config,
            }, save_path)
            print(f"  ✓ New best WGA: {best_wga*100:.2f}%")

        # Periodic checkpoint (for resuming after Colab disconnects)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(config.checkpoint_dir, f'student_{exp_name}_latest.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'adapter_state_dict': adapter.state_dict() if adapter else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wga': best_wga,
            }, save_path)

    # Final test evaluation
    print(f"\nFinal Test Evaluation ({exp_name}):")
    test_results = compute_group_accuracies(student, loaders['test'], device)
    print_results(test_results, f"Student Test Results ({exp_name})")

    # Save final results
    results_path = os.path.join(config.checkpoint_dir, f'student_{exp_name}_results.pt')
    torch.save({
        'test_results': test_results,
        'history': logger.to_dict(),
        'config': vars(config) if hasattr(config, '__dict__') else config,
    }, results_path)

    return student, logger.to_dict(), test_results


# =============================================================================
# MAIN (for running experiments)
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['teacher', 'student'],
                        help='Train teacher or student')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    # Student-specific
    parser.add_argument('--teacher_paths', nargs='+', type=str,
                        help='Paths to teacher checkpoints (for student training)')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--exp_name', type=str, default='baseline')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.mode == 'teacher':
        config = Config(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            epochs=100,
            lr=0.001,
            weight_decay=1e-3,
        )
        model, history = train_teacher(config)
        
    else:  # student
        if not args.teacher_paths:
            raise ValueError("Must provide --teacher_paths for student training")
        
        # Load teachers
        teachers = []
        for path in args.teacher_paths:
            teacher = get_teacher_model('resnet50', 2, pretrained=False)
            load_checkpoint(teacher, path)
            teachers.append(teacher)
        print(f"Loaded {len(teachers)} teachers")
        
        config = Config(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            epochs=30,
        )
        model, history, results = train_student(config, teachers, args.exp_name)
