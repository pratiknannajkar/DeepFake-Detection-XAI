"""
DeepShield AI — Image Utility Functions
Handles image loading, resizing, format conversion, and base64 encoding.
"""

import base64
import io
import os
import tempfile
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes into an OpenCV BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid image.")
    return img


def load_pil_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes into a PIL Image."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def resize_image(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image so its longest side is at most max_size, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cv2_to_base64(img: np.ndarray, fmt: str = ".png") -> str:
    """Convert an OpenCV image (BGR) to a base64-encoded string."""
    success, buffer = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer).decode("utf-8")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def apply_heatmap_overlay(
    base_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a single-channel heatmap onto a base image with a colormap."""
    # Normalize heatmap to 0-255
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

    # Resize heatmap to match base image
    heatmap_color = cv2.resize(heatmap_color, (base_img.shape[1], base_img.shape[0]))

    # Blend
    overlay = cv2.addWeighted(base_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def create_temp_jpeg(img_pil: Image.Image, quality: int = 90) -> str:
    """Save a PIL image as a temporary JPEG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img_pil.save(tmp.name, "JPEG", quality=quality)
    tmp.close()
    return tmp.name


def cleanup_temp_file(path: str):
    """Remove a temporary file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of a cv2 image."""
    h, w = img.shape[:2]
    return w, h


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
