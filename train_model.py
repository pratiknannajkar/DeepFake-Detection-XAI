"""
DeepShield AI -- EfficientNet-B0 Deepfake Detection Training Pipeline

End-to-end training script:
  1. Load dataset (Train/Validation/Test with Real/Fake subfolders)
  2. Apply augmentation (train only)
  3. Phase 1: Freeze backbone, train classifier head (3 epochs)
  4. Phase 2: Unfreeze last blocks, fine-tune (7 epochs)
  5. Evaluate on test set (accuracy, precision, recall, F1)
  6. Save best model to backend/models/efficientnet_deepfake.pth

Usage:
    python train_model.py

Requires: torch, torchvision, Pillow, scikit-learn
"""

import os
import sys
import time
import copy
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "backend", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "efficientnet_deepfake.pth")

# Training params
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4       # DataLoader workers (set to 0 on Windows if issues)
PHASE1_EPOCHS = 3     # Freeze backbone, train head only
PHASE2_EPOCHS = 7     # Unfreeze last blocks, fine-tune
PHASE1_LR = 1e-3      # Higher LR for head-only training
PHASE2_LR = 1e-5      # Low LR for fine-tuning
PATIENCE = 3           # Early stopping patience

# Classes: ImageFolder sorts alphabetically → Fake=0, Real=1
# We'll verify and remap so that: 0=Real, 1=Fake
CLASS_NAMES = ["Real", "Fake"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Get train and val/test transforms."""

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def load_datasets(train_transform, val_transform, max_samples=None):
    """Load train, validation, and test datasets.
    
    Args:
        max_samples: If set, limits each split to this many samples (for fast CPU training).
    """

    train_dir = os.path.join(DATASET_DIR, "Train")
    val_dir = os.path.join(DATASET_DIR, "Validation")
    test_dir = os.path.join(DATASET_DIR, "Test")

    # Verify directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            print(f"ERROR: Directory not found: {d}")
            sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    class_to_idx = train_dataset.class_to_idx

    # Print class mapping (ImageFolder sorts alphabetically)
    print(f"\nClass mapping: {class_to_idx}")
    print(f"  -> Fake={class_to_idx.get('Fake', '?')}, "
          f"Real={class_to_idx.get('Real', '?')}")

    print(f"\nFull dataset sizes:")
    print(f"  Train:      {len(train_dataset):>8,} images")
    print(f"  Validation: {len(val_dataset):>8,} images")
    print(f"  Test:       {len(test_dataset):>8,} images")

    # Subsample if max_samples is set (for fast CPU training)
    if max_samples and max_samples > 0:
        def make_balanced_subset(dataset, n):
            """Create a balanced subset with equal real/fake samples."""
            indices_by_class = {}
            for idx, (_, label) in enumerate(dataset.samples):
                indices_by_class.setdefault(label, []).append(idx)
            
            selected = []
            per_class = n // len(indices_by_class)
            for label, indices in indices_by_class.items():
                np.random.seed(42)  # Reproducible
                chosen = np.random.choice(indices, min(per_class, len(indices)), replace=False)
                selected.extend(chosen.tolist())
            
            np.random.shuffle(selected)
            return Subset(dataset, selected)

        train_dataset = make_balanced_subset(train_dataset, max_samples)
        val_dataset = make_balanced_subset(val_dataset, max_samples // 4)
        # Keep full test set for accurate evaluation

        print(f"\n  [SUBSET MODE] Using max {max_samples} train samples")
        print(f"  Train subset:  {len(train_dataset):>8,} images")
        print(f"  Val subset:    {len(val_dataset):>8,} images")
        print(f"  Test (full):   {len(test_dataset):>8,} images")

    print(f"  Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset):>8,} images")

    # DataLoaders
    workers = NUM_WORKERS
    if sys.platform == "win32":
        workers = 0
        print(f"\n  [Windows] Using num_workers=0 for DataLoader compatibility")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    return class_to_idx, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(num_classes=2):
    """Build EfficientNet-B0 with pretrained ImageNet weights."""

    print(f"\nLoading EfficientNet-B0 with ImageNet pretrained weights...")
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head
    # Original: (dropout, Linear(1280, 1000))
    # New:      (dropout, Linear(1280, 2))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    print(f"  Backbone: EfficientNet-B0 (1280 features)")
    print(f"  Classifier head: Dropout(0.3) -> Linear(1280, {num_classes})")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")

    return model.to(DEVICE)


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  [FREEZE] Trainable: {trainable:,} / {total:,} parameters")


def unfreeze_last_blocks(model, num_blocks=3):
    """Unfreeze the last N blocks of the backbone for fine-tuning."""
    # EfficientNet features is a Sequential of blocks
    # Unfreeze the last `num_blocks` blocks + classifier
    features = list(model.features.children())
    total_blocks = len(features)

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N blocks
    for block in features[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Always unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  [UNFREEZE] Last {num_blocks} blocks + classifier")
    print(f"  Trainable: {trainable:,} / {total:,} parameters")


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs, phase_name):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(labels).sum().item()
        running_total += labels.size(0)

        # Progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            batch_acc = 100.0 * running_correct / running_total
            print(f"    [{phase_name}] Epoch {epoch}/{total_epochs} "
                  f"Batch {batch_idx+1}/{len(loader)} "
                  f"Loss: {running_loss/running_total:.4f} "
                  f"Acc: {batch_acc:.1f}% "
                  f"({elapsed:.0f}s)")

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    elapsed = time.time() - start_time

    print(f"  [{phase_name}] Epoch {epoch}/{total_epochs} -- "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% ({elapsed:.0f}s)")

    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    """Validate / evaluate on a dataset."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            running_correct += preds.eq(labels).sum().item()
            running_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def evaluate_test(model, test_loader, criterion, class_to_idx):
    """Full evaluation on test set with metrics."""
    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*60}")

    test_loss, test_acc, preds, labels = validate(model, test_loader, criterion)

    print(f"\n  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Per-class metrics
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        # Map indices to names
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

        print(f"\n  Classification Report:")
        report = classification_report(labels, preds, target_names=target_names, digits=4)
        print(report)

        print(f"  Confusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print(f"                Predicted")
        print(f"                {target_names[0]:>8s}  {target_names[1]:>8s}")
        print(f"  Actual {target_names[0]:>5s}   {cm[0][0]:>8d}  {cm[0][1]:>8d}")
        print(f"  Actual {target_names[1]:>5s}   {cm[1][0]:>8d}  {cm[1][1]:>8d}")

    except ImportError:
        print("  (Install scikit-learn for detailed metrics: pip install scikit-learn)")

    return test_acc


def save_model(model, class_to_idx, test_acc):
    """Save the trained model and metadata."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Save model weights + metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "num_classes": 2,
        "image_size": IMAGE_SIZE,
        "architecture": "efficientnet_b0",
        "test_accuracy": test_acc,
    }

    torch.save(checkpoint, MODEL_SAVE_PATH)
    file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\n  Model saved to: {MODEL_SAVE_PATH}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Also save config as JSON for reference
    config = {
        "architecture": "efficientnet_b0",
        "num_classes": 2,
        "image_size": IMAGE_SIZE,
        "class_to_idx": class_to_idx,
        "test_accuracy": round(test_acc, 2),
        "model_file": "efficientnet_deepfake.pth",
    }
    config_path = os.path.join(MODEL_SAVE_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="DeepShield AI -- EfficientNet-B0 Training")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples (for fast CPU training). E.g. 5000")
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS,
                        help=f"Phase 1 epochs (default: {PHASE1_EPOCHS})")
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS,
                        help=f"Phase 2 epochs (default: {PHASE2_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    batch_size = args.batch_size

    print("=" * 60)
    print("  DeepShield AI -- EfficientNet-B0 Training Pipeline")
    print("=" * 60)
    print(f"\n  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print(f"  [!] No GPU detected -- training will be slower on CPU")
        if not args.max_samples:
            print(f"  [!] TIP: Use --max-samples 5000 for faster CPU training")

    # -- Load data --
    train_transform, val_transform = get_transforms()
    (class_to_idx, train_dataset, val_dataset, test_dataset,
     train_loader, val_loader, test_loader) = load_datasets(
        train_transform, val_transform, max_samples=args.max_samples
    )

    # -- Build model --
    model = build_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()

    p1_epochs = args.phase1_epochs
    p2_epochs = args.phase2_epochs

    # ================================================================
    #  PHASE 1: Freeze backbone, train classifier head only
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Train classifier head (backbone frozen)")
    print(f"  Epochs: {p1_epochs}, LR: {PHASE1_LR}")
    print(f"{'='*60}")

    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=1e-4
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, p1_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            epoch, p1_epochs, "Phase1"
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        print(f"  [Phase1] Epoch {epoch}/{p1_epochs} -- "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  * New best: {val_acc:.2f}%")

    # Load best phase 1 weights
    model.load_state_dict(best_model_wts)
    print(f"\n  Phase 1 best val accuracy: {best_val_acc:.2f}%")

    # ================================================================
    #  PHASE 2: Unfreeze last blocks, fine-tune
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Fine-tune (last 3 blocks unfrozen)")
    print(f"  Epochs: {p2_epochs}, LR: {PHASE2_LR}")
    print(f"{'='*60}")

    unfreeze_last_blocks(model, num_blocks=3)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    patience_counter = 0

    for epoch in range(1, p2_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            epoch, p2_epochs, "Phase2"
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        print(f"  [Phase2] Epoch {epoch}/{p2_epochs} -- "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  * New best: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [X] Early stopping after {PATIENCE} epochs without improvement")
                break

    # Load best overall weights
    model.load_state_dict(best_model_wts)
    print(f"\n  Overall best val accuracy: {best_val_acc:.2f}%")

    # ================================================================
    #  EVALUATE on test set
    # ================================================================
    test_acc = evaluate_test(model, test_loader, criterion, class_to_idx)

    # ================================================================
    #  SAVE model
    # ══════════════════════════════════════════════════════════════════════
    save_model(model, class_to_idx, test_acc)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Test accuracy:            {test_acc:.2f}%")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
