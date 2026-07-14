"""
SigLIP2 Multi-View Fine-tuning (V2: Patch-Level Cross-View Attention Pooling)
==============================================================================

Architecture:
  - SigLIP2 completely frozen (no gradient, pure feature extractor)
  - Uses patch-level tokens (last_hidden_state) instead of pooler_output
  - CrossViewAttentionPooler: learnable queries cross-attend to 1024 patch tokens
  - SupConLoss only (no text, no SigLipLoss)

Following PaliGemma/pi0/HiROBOT pattern:
  - Freeze SigLIP, adapt downstream
  - Preserve full spatial information from patch tokens
  - View position embeddings for camera-aware fusion

Date: 2026-06-15
"""

import os
import sys
import re
import json
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Allow launching this script from any working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from siglip2_trainer.multiview_config import MultiViewConfig
from siglip2_trainer.multiview_models import (
    MultiViewSimpleDataset,
    MultiViewSigLIPModel,
    CrossViewAttentionPooler,
    setup_multiview_model,
    load_multiview_checkpoint,
    make_multiview_collate_fn,
    split_image_to_views,
)
from siglip2_trainer.multiview_video_dataset import MultiViewVideoDataset
from siglip2_trainer.losses import SupConLoss
from siglip2_trainer.checkpoints import CheckpointManager, EarlyStopping
from siglip2_trainer.visualization import plot_loss_curve, save_training_summary
from siglip2_trainer.logging_utils import (
    NumpyEncoder, save_training_config,
    collect_dataset_info, collect_model_info,
)
from siglip2_trainer.evaluation import (
    plot_confusion_matrix,
    plot_classification_results,
)


# =============================================================================
# Multi-View Class Center Computation
# =============================================================================

def compute_multiview_class_centers(model, processor, train_dataset, config, epoch):
    """
    Compute class center vectors from fused multi-view embeddings.

    For each training image:
      1. Split into 4 views
      2. Encode + fuse via model.encode_views()
      3. Accumulate per-class

    Returns:
        class_centers: dict {class_name: center_vector (numpy)}
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"[Multi-View Class Centers] Epoch {epoch}")
    print(f"{'='*60}")

    class_list = sorted(
        train_dataset.classes,
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )

    # Group samples by class. Video samples are decoded lazily by the dataset.
    class_samples = {c: [] for c in class_list}
    for sample in train_dataset.samples:
        class_samples[sample['class']].append(sample)

    print(f"Classes: {len(class_list)}, Training samples: {len(train_dataset.samples)}")

    class_centers = {}
    batch_size = 8

    with torch.no_grad():
        for category in class_list:
            category_samples = class_samples[category]
            if len(category_samples) == 0:
                print(f"  Warning: no samples for {category}")
                continue

            print(f"  Processing {category}: {len(category_samples)} samples")

            features_list = []
            for i in range(0, len(category_samples), batch_size):
                batch_samples = category_samples[i:i + batch_size]
                batch_views = []

                for sample in batch_samples:
                    try:
                        if 'image_path' in sample:
                            with Image.open(sample['image_path']) as img:
                                img = img.convert('RGB')
                                img.load()
                            views = split_image_to_views(img)
                        else:
                            views = train_dataset.load_views(
                                sample, apply_augmentation=False
                            )
                        batch_views.extend(views)
                    except Exception as e:
                        print(f"    Warning: failed to load sample: {e}")
                        continue

                if len(batch_views) == 0:
                    continue

                inputs = processor(images=batch_views, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(config.DEVICE)

                fused_embeds = model.encode_views(pixel_values)  # [B, D]
                features_list.append(fused_embeds.cpu().numpy())

            if len(features_list) > 0:
                features = np.vstack(features_list)
                center = np.mean(features, axis=0)
                center = center / (np.linalg.norm(center) + 1e-12)
                class_centers[category] = center
                print(f"    {category} center computed (dim: {center.shape})")

    # Save to graph_info.json
    _save_graph_info(train_dataset, class_centers, config, epoch)
    if hasattr(train_dataset, 'close'):
        train_dataset.close()

    model.train()
    return class_centers


def _save_graph_info(train_dataset, class_centers, config, epoch):
    """Copy and update graph_info.json with computed class centers."""
    original_graph_path = os.path.join(train_dataset.image_root, "graph_info.json")

    if not os.path.exists(original_graph_path):
        print(f"  Warning: graph_info.json not found at {original_graph_path}")
        return

    eval_dir = os.path.join(config.MODEL_DIR, 'classification_viz')
    os.makedirs(eval_dir, exist_ok=True)

    graph_copy_path = os.path.join(eval_dir, f'graph_info_{epoch}.json')
    shutil.copy2(original_graph_path, graph_copy_path)

    try:
        with open(graph_copy_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        nodes = graph_data.get('nodes', [])
        updated_count = 0

        # Match node ids to class-number suffixes. Video labels can skip
        # states, so mapping by sorted-list position would corrupt M5/M6.
        class_by_id = {}
        for category_name in class_centers:
            match = re.search(r'\d+', category_name)
            if match:
                class_by_id[int(match.group())] = category_name

        for node in nodes:
            node_id = node.get('node_id')
            if node_id is None:
                continue

            try:
                if isinstance(node_id, str):
                    node_id_int = int(re.search(r'\d+', node_id).group())
                else:
                    node_id_int = int(node_id)

                category_name = class_by_id.get(node_id_int)
                if category_name is not None:
                    node['center_feature_siglip2'] = class_centers[category_name].tolist()
                    updated_count += 1
                else:
                    print(f"  Warning: node {node_id} has no training samples")
            except Exception as e:
                print(f"  Warning: node {node_id} processing failed: {e}")

        print(f"  Updated {updated_count}/{len(nodes)} nodes in graph_info")

        # Save with compact feature format
        temp_data = json.loads(json.dumps(graph_data))
        for node in temp_data.get('nodes', []):
            if 'center_feature_siglip2' in node:
                node['center_feature_siglip2'] = json.dumps(
                    node['center_feature_siglip2'],
                    separators=(',', ':')
                )

        with open(graph_copy_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {graph_copy_path}")

    except Exception as e:
        print(f"  Error updating graph_info: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# Multi-View Evaluation
# =============================================================================

def evaluate_multiview_with_class_centers(model, processor, val_dataset, class_centers, config, epoch):
    """
    Evaluate using multi-view fused features vs class centers.

    For each validation sample:
      1. Split image into 4 views
      2. Encode + fuse
      3. Compute cosine similarity to all class centers
      4. Predict = argmax(similarity)
    """
    from datetime import datetime

    model.eval()

    print(f"\n{'='*60}")
    print(f"[Multi-View Evaluation] Epoch {epoch}")
    print(f"Val set: {len(val_dataset)} samples, Centers: {len(class_centers)} classes")
    print(f"{'='*60}\n")

    class_list = sorted(class_centers.keys())
    num_classes = len(class_list)

    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_list}
    per_class_total = {cls: 0 for cls in class_list}
    confusion_matrix = {(t, p): 0 for t in class_list for p in class_list}

    error_samples = []
    samples_per_class = {cls: [] for cls in class_list}
    max_samples_per_class = 3

    batch_size = 8

    with torch.no_grad():
        for start_idx in range(0, len(val_dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(val_dataset))

            batch_views = []
            batch_true_classes = []
            batch_images_for_viz = []

            for idx in range(start_idx, end_idx):
                sample = val_dataset[idx]
                views = sample['views']
                batch_views.extend(views)
                batch_true_classes.append(sample['class'])
                batch_images_for_viz.append(views[0])

            inputs = processor(images=batch_views, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(config.DEVICE)
            fused_embeds = model.encode_views(pixel_values)

            for i, (vec, true_cls) in enumerate(zip(fused_embeds.cpu(), batch_true_classes)):
                vec_np = vec.numpy()

                similarities = {}
                for cls, center in class_centers.items():
                    similarities[cls] = float(np.dot(vec_np, center))

                pred_cls = max(similarities.items(), key=lambda x: x[1])[0]

                total += 1
                per_class_total[true_cls] += 1
                confusion_matrix[(true_cls, pred_cls)] += 1

                if pred_cls == true_cls:
                    correct += 1
                    per_class_correct[true_cls] += 1
                else:
                    error_samples.append({
                        'true_class': true_cls,
                        'pred_class': pred_cls,
                        'similarities': similarities,
                        'sample_idx': start_idx + i
                    })

                if len(samples_per_class[true_cls]) < max_samples_per_class:
                    samples_per_class[true_cls].append({
                        'image': batch_images_for_viz[i],
                        'true_class': true_cls,
                        'pred_class': pred_cls,
                        'similarities': similarities,
                        'correct': pred_cls == true_cls
                    })

    # Metrics
    accuracy = correct / total if total > 0 else 0
    per_class_accuracy = {
        cls: per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
        for cls in class_list
    }

    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_class_accuracy': per_class_accuracy,
        'num_classes': num_classes
    }

    print(f"  Overall accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"\n  Per-class accuracy:")
    for cls in class_list:
        acc = per_class_accuracy[cls]
        cnt = per_class_total[cls]
        print(f"    {cls}: {acc:.1%} ({per_class_correct[cls]}/{cnt})")

    # Save visualizations
    save_dir = os.path.join(config.MODEL_DIR, 'classification_viz')
    os.makedirs(save_dir, exist_ok=True)

    visualization_samples = []
    for cls in class_list:
        visualization_samples.extend(samples_per_class[cls])

    try:
        plot_classification_results(
            visualization_samples, class_list, metrics, save_dir, epoch
        )
        print(f"\n  Classification visualization saved ({len(visualization_samples)} samples)")
    except Exception as e:
        print(f"  Warning: visualization failed: {e}")

    # Save results JSON
    result_file = os.path.join(save_dir, f'results_epoch_{epoch}.json')
    confusion_matrix_2d = []
    for true_cls in class_list:
        row = [confusion_matrix[(true_cls, pred_cls)] for pred_cls in class_list]
        confusion_matrix_2d.append(row)

    error_samples_serializable = []
    for sample in error_samples[:50]:
        error_samples_serializable.append({
            'true_class': sample['true_class'],
            'pred_class': sample['pred_class'],
            'similarities': {k: float(v) for k, v in sample['similarities'].items()},
            'sample_idx': int(sample['sample_idx'])
        })

    result_data = {
        'epoch': int(epoch) if isinstance(epoch, int) else str(epoch),
        'timestamp': datetime.now().isoformat(),
        'mode': 'multi-view V2 (patch-level cross-attention)',
        'metrics': {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total),
            'num_classes': int(num_classes),
            'per_class_accuracy': {k: float(v) for k, v in per_class_accuracy.items()}
        },
        'confusion_matrix': {
            'classes': list(class_list),
            'matrix': [[int(x) for x in row] for row in confusion_matrix_2d]
        },
        'error_analysis': {
            'total_errors': int(total - correct),
            'error_samples': error_samples_serializable
        }
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Results saved: {result_file}")

    # Confusion matrix plot
    try:
        plot_confusion_matrix(
            np.array(confusion_matrix_2d), class_list, save_dir, epoch
        )
    except Exception as e:
        print(f"  Warning: confusion matrix plot failed: {e}")

    if hasattr(val_dataset, 'close'):
        val_dataset.close()
    model.train()
    print(f"{'='*60}\n")

    return metrics


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, supcon_fn, optimizer, dataloader, epoch, config):
    model.train()
    epoch_losses = []
    current_lr = optimizer.param_groups[0]['lr']
    accum_steps = config.GRADIENT_ACCUMULATION_STEPS

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(config.DEVICE)  # [4B, C, H, W]
        labels = batch['labels'].to(config.DEVICE)

        # Forward pass (SigLIP frozen inside encode_views via torch.no_grad)
        image_embeds = model.encode_views(pixel_values)  # [B, D]

        # SupConLoss only
        loss = supcon_fn(image_embeds, labels)

        # Gradient accumulation
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        loss_value = loss.item() * accum_steps
        epoch_losses.append(loss_value)

        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch:3d}][Batch {batch_idx:3d}] "
                  f"SupCon: {loss_value:.4f}  LR: {current_lr:.2e}")

    return epoch_losses


# =============================================================================
# Main
# =============================================================================

def main():
    config = MultiViewConfig()

    print("=" * 60)
    print("SigLIP2 Multi-View Fine-tuning V2 (Patch-Level Cross-Attention)")
    print("=" * 60)
    print(f"Model:           {config.SIGLIP_MODEL}")
    print(f"Views:           {config.NUM_VIEWS} ({', '.join(config.VIEW_NAMES)})")
    print(f"Epochs:          {config.EPOCHS}")
    print(f"Batch size:      {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS})")
    print(f"Learning rate:   {config.LEARNING_RATE}")
    print(f"Query tokens:    {config.NUM_QUERY_TOKENS}")
    print(f"Pooler:          {config.POOLER_NUM_LAYERS} layers, {config.POOLER_NUM_HEADS} heads")
    print(f"Scheduler:       {config.SCHEDULER_TYPE}")
    print(f"SigLIP:          completely frozen (patch-level tokens)")
    print(f"Early stopping:  {config.EARLY_STOPPING}")
    if config.EARLY_STOPPING:
        print(f"  Patience:      {config.PATIENCE}")
    print("=" * 60)
    print()

    # ===== Model Setup =====
    model, processor, optimizer, scheduler = setup_multiview_model(config)
    supcon_fn = SupConLoss(temperature=config.SUPCON_TEMPERATURE)
    print(f"SupCon Loss (temperature={config.SUPCON_TEMPERATURE})")
    print()

    # ===== Dataset =====
    print("Loading multi-view dataset...")
    if config.DATA_MODE == 'video':
        video_dataset_kwargs = dict(
            video_root=config.VIDEO_ROOT,
            label_root=config.VIDEO_LABEL_ROOT,
            val_ratio=config.VAL_RATIO,
            frame_stride=config.VIDEO_FRAME_STRIDE,
            max_samples_per_class=config.VIDEO_SAMPLES_PER_CLASS,
        )
        train_dataset = MultiViewVideoDataset(
            **video_dataset_kwargs,
            split='train',
            use_augmentation=config.USE_AUGMENTATION,
            augmentation_config=config.AUGMENTATION_CONFIG,
        )
        val_dataset = MultiViewVideoDataset(
            **video_dataset_kwargs,
            split='val',
            use_augmentation=False,
        )
        dataset_root_for_logging = config.VIDEO_ROOT
    elif config.DATA_MODE == 'image':
        train_dataset = MultiViewSimpleDataset(
            image_root=config.IMAGE_ROOT,
            text_root=None,
            use_augmentation=config.USE_AUGMENTATION,
            augmentation_config=config.AUGMENTATION_CONFIG,
            is_training=True,
            val_ratio=config.VAL_RATIO,
            test_ratio=0.0,
            split='train',
        )
        val_dataset = MultiViewSimpleDataset(
            image_root=config.IMAGE_ROOT,
            text_root=None,
            use_augmentation=False,
            is_training=False,
            val_ratio=config.VAL_RATIO,
            test_ratio=0.0,
            split='val',
        )
        dataset_root_for_logging = config.IMAGE_ROOT
    else:
        raise ValueError(f"Unsupported DATA_MODE: {config.DATA_MODE!r}")

    # The dataset is the source of truth for the number of trainable states.
    train_classes = sorted(train_dataset.classes)
    val_classes = sorted(val_dataset.classes)
    if train_classes != val_classes:
        raise ValueError(
            "训练集和验证集类别不一致:\n"
            f"  train-only: {sorted(set(train_classes) - set(val_classes))}\n"
            f"  val-only:   {sorted(set(val_classes) - set(train_classes))}"
        )
    detected_num_classes = len(train_classes)
    if getattr(config, 'NUM_CLASSES', None) != detected_num_classes:
        print(
            f"[Class count] 配置 NUM_CLASSES={getattr(config, 'NUM_CLASSES', None)}，"
            f"但数据集检测到 {detected_num_classes} 个类别，自动修正。"
        )
        config.NUM_CLASSES = detected_num_classes
    print(f"[Class count] 已确认 {config.NUM_CLASSES} 个状态: {train_classes}")

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Split: train {1-config.VAL_RATIO:.0%} / val {config.VAL_RATIO:.0%}")
    print(f"  Eval interval: every {config.EVAL_EVERY_N_EPOCHS} epochs")
    print()

    collate_fn = make_multiview_collate_fn(processor, config)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=(config.DEVICE != "cpu"),
        collate_fn=collate_fn,
        num_workers=4,
    )
    print()

    # ===== Save training config =====
    print("Collecting training configuration...")
    dataset_info = collect_dataset_info(train_dataset, dataset_root_for_logging, None)
    dataset_info['mode'] = 'multi-view V2 (patch-level cross-attention)'
    dataset_info['views_per_sample'] = config.NUM_VIEWS
    model_info = collect_model_info(model)
    save_training_config(config, dataset_info, model_info, config.MODEL_DIR)
    print()

    # ===== Checkpoint & Early Stopping =====
    checkpoint_manager = CheckpointManager(
        model_dir=config.MODEL_DIR,
        model_name=config.MODEL_NAME,
        max_checkpoints=config.MAX_CHECKPOINTS,
    )

    early_stopping = None
    if config.EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.PATIENCE, min_delta=config.MIN_DELTA, mode='max'
        )
        print(f"Early stopping enabled (patience={config.PATIENCE} evaluations, "
              f"min_delta={config.MIN_DELTA}, tracking eval accuracy)")

    print(f"Checkpoint: every {config.SAVE_EVERY_N_EPOCHS} epochs, max {config.MAX_CHECKPOINTS}")

    # ===== Resume from checkpoint =====
    start_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = None
    best_eval_accuracy = None
    best_eval_epoch = None

    if config.RESUME_FROM_CHECKPOINT:
        try:
            resume_info = load_multiview_checkpoint(
                config.RESUME_FROM_CHECKPOINT, model, optimizer, scheduler
            )
            start_epoch = resume_info.get('start_epoch', 0)
            if resume_info.get('previous_loss') is not None:
                best_loss = resume_info['previous_loss']
            print(f"\nResumed from checkpoint")
        except Exception as e:
            print(f"\nCheckpoint load failed: {e}")
            print(f"Starting from scratch")
            start_epoch = 0

    # ===== Training Loop =====
    print("-" * 60)
    if start_epoch > 0:
        print(f"Resuming training (from Epoch {start_epoch})")
    else:
        print("Starting training...")
    print("-" * 60)

    start_time = time.time()
    loss_history = []
    lr_history = []
    accuracy_history = []
    accuracy_epochs = []
    best_loss_updated = False

    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_losses = train_one_epoch(
            model, supcon_fn, optimizer,
            dataloader, epoch, config
        )
        loss_history.extend(epoch_losses)

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if scheduler is not None:
            scheduler.step()

        avg_loss = np.mean(epoch_losses)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_loss_epoch = epoch
            best_loss_updated = True
            checkpoint_manager.save_best_loss(model, optimizer, epoch, avg_loss, current_lr, scheduler)

        _cached_centers = None
        _cached_centers_epoch = None

        _skip_early = epoch < getattr(config, 'EVAL_SKIP_FIRST_N', 0)

        # Checkpoint save + class center computation
        if not _skip_early and ((epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0 or epoch == 0):
            print(f"\n[Checkpoint]")
            checkpoint_manager.save(model, optimizer, epoch, avg_loss, current_lr, scheduler)
            try:
                _cached_centers = compute_multiview_class_centers(
                    model, processor, train_dataset, config, epoch
                )
                _cached_centers_epoch = epoch
                print(f"[Centers] Saved graph_info_{epoch}.json")
            except Exception as e:
                import traceback
                print(f"[Centers] Computation failed: {e}\n{traceback.format_exc()}")

            if best_loss_updated and is_best and _cached_centers is not None:
                src = os.path.join(config.MODEL_DIR, 'classification_viz',
                                   f'graph_info_{epoch}.json')
                dst = os.path.join(config.MODEL_DIR, 'classification_viz',
                                   'graph_info_best_loss.json')
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"[Centers] graph_info_best_loss.json copied from epoch {epoch}")
                best_loss_updated = False

        marker = " BEST_LOSS" if is_best else ""
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.2e}{marker}")
        print("-" * 60)

        # Evaluation
        if config.ENABLE_EVAL and not _skip_early and (epoch + 1) % config.EVAL_EVERY_N_EPOCHS == 0:
            try:
                print(f"\nMulti-view evaluation (Epoch {epoch})...")

                if _cached_centers_epoch == epoch and _cached_centers is not None:
                    class_centers = _cached_centers
                    print(f"[Centers] Reusing cached centers from this epoch")
                else:
                    class_centers = compute_multiview_class_centers(
                        model, processor, train_dataset, config, epoch
                    )

                metrics = evaluate_multiview_with_class_centers(
                    model, processor, val_dataset, class_centers, config, epoch
                )

                print(f"Evaluation done (accuracy: {metrics['accuracy']:.1%})")

                accuracy_history.append(metrics['accuracy'])
                accuracy_epochs.append(epoch)

                if (best_eval_accuracy is None or
                        metrics['accuracy'] > best_eval_accuracy + config.MIN_DELTA):
                    best_eval_accuracy = metrics['accuracy']
                    best_eval_epoch = epoch

                if early_stopping is not None:
                    should_stop = early_stopping(metrics['accuracy'], epoch)
                    if should_stop:
                        best_score, best_epoch = early_stopping.get_best_info()
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered!")
                        print(f"Best accuracy: {best_score:.1%} at epoch {best_epoch}")
                        print(f"{'='*60}\n")
                        break
                    if early_stopping.counter > 0:
                        print(f"  Early stopping counter: {early_stopping.counter}/{config.PATIENCE}")
                    else:
                        print(f"  New best accuracy: {early_stopping.best_score:.1%}!")
                        checkpoint_manager.save_best_eval(
                            model, optimizer, epoch, metrics['accuracy'],
                            avg_loss, current_lr, scheduler
                        )

            except Exception as e:
                import traceback
                print(f"Evaluation failed: {e}")
                print(f"Details:\n{traceback.format_exc()}")

        _val_acc = accuracy_history[-1] if accuracy_epochs and accuracy_epochs[-1] == epoch else None
        plot_loss_curve(loss_history, lr_history, config, epoch, avg_loss,
                       accuracy_history, accuracy_epochs, val_accuracy=_val_acc)

    # ===== Training Complete =====
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training completed!")
    print(f"Total time:   {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best loss:    {best_loss:.4f}")
    print(f"Best loss epoch: {best_loss_epoch}")
    print(f"Final loss:   {loss_history[-1]:.4f}")
    if accuracy_history:
        print(f"Best accuracy: {max(accuracy_history):.2%}")
        print(f"Best accuracy epoch: {best_eval_epoch}")
        print(f"Final accuracy: {accuracy_history[-1]:.2%}")
    print("=" * 60)
    print(f"\nSaved checkpoints: {checkpoint_manager.get_saved_epochs()}")

    # ===== Save final graph_info for best models =====
    print(f"\n{'='*60}")
    print(f"[Saving best model graph_info.json]")
    print(f"{'='*60}")

    best_loss_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_best_loss.pt")
    if os.path.exists(best_loss_path):
        print(f"\nProcessing graph_info_best_loss.json...")
        try:
            checkpoint = torch.load(best_loss_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(config.DEVICE)
            print(f"  Loaded: epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")

            compute_multiview_class_centers(
                model, processor, train_dataset, config, 'best_loss'
            )
        except Exception as e:
            print(f"  Failed: {e}")

    best_eval_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_best_eval.pt")
    if os.path.exists(best_eval_path) and accuracy_history:
        print(f"\nProcessing graph_info_best_eval.json...")
        try:
            checkpoint = torch.load(best_eval_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(config.DEVICE)
            print(f"  Loaded: epoch={checkpoint['epoch']}, accuracy={checkpoint['eval_accuracy']:.2%}")

            compute_multiview_class_centers(
                model, processor, train_dataset, config, 'best_eval'
            )
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\n{'='*60}")
    print("All graph_info.json files saved")
    print(f"{'='*60}")

    plot_loss_curve(loss_history, lr_history, config,
                    current_epoch=epoch, current_loss=loss_history[-1],
                    accuracy_history=accuracy_history, accuracy_epochs=accuracy_epochs,
                    val_accuracy=accuracy_history[-1] if accuracy_history else None)
    save_training_summary(loss_history, lr_history, config, epoch,
                         accuracy_history, accuracy_epochs)

    print("\nDone!")


if __name__ == "__main__":
    main()
