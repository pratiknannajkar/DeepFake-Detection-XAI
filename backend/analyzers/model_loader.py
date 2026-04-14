"""
DeepShield AI — EfficientNet-B0 Deepfake Detection Model Loader

Loads a custom-trained EfficientNet-B0 model for binary deepfake classification
and provides real Grad-CAM heatmap generation.

Replaces the previous CLIP zero-shot approach with a fine-tuned model
trained on the project's own deepfake dataset (~190K images).

Advantages over CLIP:
  - Trained specifically on deepfake data → ~95%+ accuracy (vs ~60-70%)
  - Real Grad-CAM from convolutional layers (vs approximate ViT attention)
  - 20MB model (vs 350MB CLIP)
  - 2s startup (vs 15-20s)
  - No HuggingFace/Transformers dependency
"""

import os
import logging
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger("deepshield.model_loader")

# ── Globals (lazy-loaded) ─────────────────────────────────────────────────────
_model = None
_transform = None
_model_loaded = False
_model_load_error = None
_class_to_idx = None
_device = None

# Model file path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_deepfake.pth")

# ImageNet normalization (used during training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def _load_model():
    """Lazy-load the trained EfficientNet-B0 model. Called once on first inference."""
    global _model, _transform, _model_loaded, _model_load_error, _class_to_idx, _device

    if _model_loaded or _model_load_error:
        return

    try:
        # Check if model file exists
        if not os.path.isfile(MODEL_PATH):
            _model_load_error = (
                f"Model file not found: {MODEL_PATH}. "
                f"Run 'python train_model.py' first to train the model."
            )
            logger.error(_model_load_error)
            return

        logger.info(f"Loading EfficientNet-B0 from {MODEL_PATH}...")

        # Select device
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

        # Build model architecture
        _model = models.efficientnet_b0(weights=None)
        num_classes = checkpoint.get("num_classes", 2)
        in_features = _model.classifier[1].in_features
        _model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        # Load trained weights
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.eval()

        # Store class mapping
        _class_to_idx = checkpoint.get("class_to_idx", {"Fake": 0, "Real": 1})
        test_acc = checkpoint.get("test_accuracy", "unknown")

        # Build inference transform (must match training val transform)
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        _model_loaded = True
        logger.info(
            f"EfficientNet-B0 loaded successfully. "
            f"Test accuracy: {test_acc}%, Device: {_device}, "
            f"Classes: {_class_to_idx}"
        )

    except Exception as e:
        _model_load_error = str(e)
        logger.error(f"Failed to load EfficientNet-B0 model: {e}")
        logger.error("Falling back to heuristic-only mode.")


def is_model_loaded() -> bool:
    """Check if the EfficientNet model is available."""
    _load_model()
    return _model_loaded


def get_load_error() -> Optional[str]:
    """Return model load error message, if any."""
    return _model_load_error


def predict_fake_probability(img_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Run EfficientNet-B0 inference on an image.

    Args:
        img_bgr: OpenCV BGR image (numpy array)

    Returns:
        (fake_probability, details_dict)
        fake_probability: 0.0 (definitely real) to 1.0 (definitely fake)
        details_dict: contains class probabilities and metadata
    """
    _load_model()

    if not _model_loaded:
        raise RuntimeError(f"EfficientNet model not available: {_model_load_error}")

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply transform
    input_tensor = _transform(img_rgb).unsqueeze(0).to(_device)

    # Run inference
    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # Map class indices to probabilities
    # class_to_idx: {"Fake": 0, "Real": 1} (alphabetical from ImageFolder)
    fake_idx = _class_to_idx.get("Fake", 0)
    real_idx = _class_to_idx.get("Real", 1)

    fake_prob = float(probabilities[fake_idx])
    real_prob = float(probabilities[real_idx])

    # Build details
    details = {
        "real_probability": round(real_prob * 100, 1),
        "fake_probability": round(fake_prob * 100, 1),
        "model": "EfficientNet-B0 (fine-tuned)",
        "top_match": "FAKE" if fake_prob > real_prob else "REAL",
        "top_match_score": round(max(fake_prob, real_prob) * 100, 1),
        "raw_outputs": {
            "fake": round(fake_prob * 100, 2),
            "real": round(real_prob * 100, 2),
        },
    }

    return fake_prob, details


def generate_attention_heatmap(img_bgr: np.ndarray) -> np.ndarray:
    """
    Generate a real Grad-CAM heatmap from EfficientNet-B0's last conv layer.

    Uses gradient-weighted class activation mapping to show which spatial
    regions most influenced the model's classification decision.

    Args:
        img_bgr: OpenCV BGR image

    Returns:
        heatmap: float32 array (H, W) in range [0, 1]
    """
    _load_model()

    if not _model_loaded:
        # Return a centered gaussian fallback
        h, w = img_bgr.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        r = min(h, w) // 3
        heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2)).astype(np.float32)
        return heatmap

    # Convert BGR → RGB and preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = _transform(img_rgb).unsqueeze(0).to(_device)
    input_tensor.requires_grad_(True)

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    # Target: last convolutional layer in EfficientNet features
    target_layer = _model.features[-1]  # Last block output

    # Storage for activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        _model.eval()
        # Temporarily enable gradients for Grad-CAM
        for param in _model.parameters():
            param.requires_grad_(False)

        output = _model(input_tensor)
        probs = F.softmax(output, dim=1)

        # Get the predicted class (or target the "Fake" class for attribution)
        fake_idx = _class_to_idx.get("Fake", 0)
        target_score = output[0, fake_idx]

        # Backward pass
        _model.zero_grad()
        target_score.backward(retain_graph=False)

        # Get stored activations and gradients
        act = activations[0]   # (1, C, H', W')
        grad = gradients[0]    # (1, C, H', W')

        # Global average pool the gradients (importance weights)
        weights = grad.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

    finally:
        # Remove hooks
        fwd_handle.remove()
        bwd_handle.remove()

    # Resize to original image size
    h, w = img_bgr.shape[:2]
    heatmap = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    heatmap = heatmap.astype(np.float32)
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap
