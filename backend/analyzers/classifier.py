"""
DeepShield AI — CNN Classifier with Grad-CAM Explainability
Provides a deep-learning classification layer with visual explanations.

NOTE: This module is structured and ready for a real pre-trained model
(e.g., XceptionNet or EfficientNet fine-tuned on FaceForensics++).
Without a model file, it returns simulated results based on the other
analyzer scores for demonstration purposes.
"""

import numpy as np
import cv2
import os
from typing import Dict, Any, Optional

from backend.utils.image_utils import cv2_to_base64, load_image_from_bytes, apply_heatmap_overlay


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ClassifierAnalyzer:
    """
    CNN-based deepfake classifier with Grad-CAM explainability.

    In demo mode (no model loaded), synthesizes a classification result
    from ELA/DCT/Forensics scores and generates a simulated Grad-CAM
    heatmap highlighting the face region.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load a pre-trained model from the models directory."""
        model_path = os.path.join(MODEL_DIR, "deepfake_detector.h5")
        alt_path = os.path.join(MODEL_DIR, "deepfake_detector.pth")

        if os.path.exists(model_path):
            try:
                # TensorFlow/Keras model
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                self.framework = "tensorflow"
            except Exception:
                pass
        elif os.path.exists(alt_path):
            try:
                # PyTorch model
                import torch
                self.model = torch.load(alt_path, map_location="cpu")
                self.model.eval()
                self.model_loaded = True
                self.framework = "pytorch"
            except Exception:
                pass

    def _generate_simulated_gradcam(
        self, img: np.ndarray, focus_regions: Optional[list] = None
    ) -> np.ndarray:
        """
        Generate a simulated Grad-CAM heatmap for demonstration.
        Creates a Gaussian-based activation map focused on the face region.
        """
        h, w = img.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if focus_regions:
            # Focus on specific detected regions
            for region in focus_regions:
                cx, cy = region.get("cx", w // 2), region.get("cy", h // 2)
                radius = region.get("radius", min(h, w) // 4)
                intensity = region.get("intensity", 1.0)

                y_grid, x_grid = np.ogrid[:h, :w]
                gaussian = np.exp(
                    -((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                    / (2 * (radius ** 2))
                ).astype(np.float32)
                heatmap += gaussian * intensity
        else:
            # Default: focused on center (where face typically is)
            cx, cy = w // 2, h // 2
            y_grid, x_grid = np.ogrid[:h, :w]
            gaussian = np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                / (2 * ((min(h, w) // 3) ** 2))
            ).astype(np.float32)
            heatmap = gaussian

        # Normalize to 0-1 range
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def _compute_focus_regions(self, forensics_result: Optional[dict]) -> list:
        """
        Determine which regions to highlight based on forensics findings.

        FORENSIC HIERARCHY: Prioritize boundary regions (chin, hairline)
        over central features (mouth). This prevents the Grad-CAM from
        being hyper-focused on the lower face where lighting and compression
        naturally create false attention.
        """
        regions = []

        if not forensics_result or not forensics_result.get("face_detected"):
            return regions

        bbox = forensics_result.get("face_bbox")
        if not bbox:
            return regions

        x, y, w, h = bbox
        face_cx, face_cy = x + w // 2, y + h // 2

        # PRIORITY 1: Boundary artifacts (chin, hairline, ears)
        # These are the strongest blending indicators for deepfakes
        if forensics_result.get("boundaries", {}).get("is_suspicious"):
            # Chin region
            regions.append({
                "cx": face_cx, "cy": y + h,
                "radius": w // 2, "intensity": 1.0
            })
            # Hairline region
            regions.append({
                "cx": face_cx, "cy": y,
                "radius": w // 2, "intensity": 0.9
            })
            # Left ear/boundary
            regions.append({
                "cx": x, "cy": face_cy,
                "radius": w // 4, "intensity": 0.7
            })
            # Right ear/boundary
            regions.append({
                "cx": x + w, "cy": face_cy,
                "radius": w // 4, "intensity": 0.7
            })

        # PRIORITY 2: Eye reflections (biological signal)
        if forensics_result.get("eye_reflections", {}).get("is_suspicious"):
            regions.append({
                "cx": face_cx, "cy": y + h // 3,
                "radius": w // 4, "intensity": 0.85
            })

        # PRIORITY 3: Mouth — ONLY if it passed the spatial threshold gate
        # (i.e., has_boundary_discontinuity is True)
        mouth_data = forensics_result.get("mouth", {})
        if mouth_data.get("is_suspicious") and mouth_data.get("has_boundary_discontinuity", False):
            regions.append({
                "cx": face_cx, "cy": y + int(h * 0.75),
                "radius": w // 5, "intensity": 0.5  # Lower intensity
            })

        # Default: whole face with slight boundary emphasis
        if not regions:
            regions.append({
                "cx": face_cx, "cy": face_cy,
                "radius": max(w, h) // 2, "intensity": 0.5
            })

        return regions

    def analyze(
        self,
        img_bytes: bytes,
        ela_score: float = 0,
        dct_score: float = 0,
        forensics_score: float = 0,
        forensics_result: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Classify the image and generate Grad-CAM explanation.

        In demo mode, aggregates other analyzer scores.

        Returns:
            dict with:
              - prediction: "FAKE" or "REAL"
              - confidence: 0-100 percentage
              - gradcam_b64: base64 Grad-CAM overlay
              - model_loaded: whether a real model was used
              - reasoning: list of factors that contributed to the decision
        """
        img_cv2 = load_image_from_bytes(img_bytes)

        if self.model_loaded:
            # Real model inference path
            return self._real_inference(img_cv2, img_bytes)
        else:
            # Demo mode: aggregate other scores
            return self._demo_inference(
                img_cv2, ela_score, dct_score, forensics_score, forensics_result
            )

    def _demo_inference(
        self,
        img: np.ndarray,
        ela_score: float,
        dct_score: float,
        forensics_score: float,
        forensics_result: Optional[dict],
    ) -> Dict[str, Any]:
        """
        Demo inference — aggregates ELA, DCT, and Forensics scores.

        Strategy (v2 — accuracy-focused):
        1. ELA and DCT are PRIMARY detection signals (they measure pixel/frequency
           anomalies that ALL AI-generated images exhibit)
        2. Forensics biometrics are SUPPORTING signals (they only help for
           face-swaps, not for fully AI-generated/illustrated images)
        3. No cross-layer FP suppression — it was hiding real detections
        4. A single strong signal (≥55) is enough for SUSPICIOUS
        5. Two agreeing signals (≥30 each) with combined≥55 is FAKE
        """

        # ── Extract forensic sub-signals ──────────────────────────────────
        boundary_suspicious = False
        eye_suspicious = False
        mouth_suspicious = False
        symmetry_suspicious = False

        if forensics_result:
            boundary_suspicious = forensics_result.get("boundaries", {}).get("is_suspicious", False)
            eye_suspicious = forensics_result.get("eye_reflections", {}).get("is_suspicious", False)
            mouth_suspicious = forensics_result.get("mouth", {}).get("is_suspicious", False)
            symmetry_suspicious = forensics_result.get("symmetry", {}).get("is_suspicious", False)

        # ── Score Aggregation — ELA/DCT Primary ───────────────────────────
        # ELA and DCT directly measure pixel-level and frequency artifacts.
        # These are present in ALL AI-generated images (illustrated, photorealistic,
        # face-swaps). Forensics biometrics only catch face-swaps.
        scores = [ela_score, dct_score, forensics_score]
        max_score = max(scores)
        ela_dct_max = max(ela_score, dct_score)

        # Weighted average: ELA/DCT dominate because they detect ALL types
        weighted_avg = (
            ela_score * 0.35
            + dct_score * 0.35
            + forensics_score * 0.30
        )

        # Boundary-specific boost: strongest face-swap indicator
        if boundary_suspicious:
            boundary_score = forensics_result.get("boundaries", {}).get("score", 0)
            weighted_avg = max(weighted_avg, boundary_score * 0.85)

        # Combined score: blend max with weighted avg
        # Give more weight to the max signal — a single strong detection
        # should not be averaged away by clean layers
        combined_score = max_score * 0.55 + weighted_avg * 0.45

        # ── Multi-signal agreement boost ──────────────────────────────────
        flagged_count = sum(1 for s in scores if s >= 30)
        ela_dct_flagged = sum(1 for s in [ela_score, dct_score] if s >= 30)

        if flagged_count >= 3:
            combined_score = min(100, combined_score + 15)
        elif flagged_count >= 2:
            combined_score = min(100, combined_score + 10)

        # ELA+DCT agreement is especially strong (pixel + frequency = clear signal)
        if ela_dct_flagged >= 2 and ela_dct_max >= 40:
            combined_score = min(100, combined_score + 8)

        # ── Biological signal corroboration ───────────────────────────────
        if boundary_suspicious and eye_suspicious:
            combined_score = min(100, combined_score + 12)
        elif boundary_suspicious or eye_suspicious:
            combined_score = min(100, combined_score + 5)

        # ── Strong single-signal override ─────────────────────────────────
        # A very strong ELA or DCT signal alone is meaningful
        if ela_dct_max >= 65:
            combined_score = max(combined_score, ela_dct_max * 0.90)
        elif ela_dct_max >= 50:
            combined_score = max(combined_score, ela_dct_max * 0.80)
        elif ela_dct_max >= 40:
            combined_score = max(combined_score, ela_dct_max * 0.70)

        combined_score = min(100, max(0, round(combined_score, 1)))

        # ── Prediction ────────────────────────────────────────────────────
        # Thresholds:
        #   0-29  = REAL     (all layers clean)
        #   30-49 = SUSPICIOUS (some signals but not conclusive)
        #   50+   = FAKE    (strong signal from at least one layer)

        if combined_score >= 60 and flagged_count >= 2:
            prediction = "FAKE"
            confidence = min(99, combined_score + 15)
        elif combined_score >= 55 and ela_dct_flagged >= 2:
            prediction = "FAKE"
            confidence = min(95, combined_score + 10)
        elif combined_score >= 50:
            prediction = "FAKE"
            confidence = min(90, combined_score + 5)
        elif combined_score >= 35:
            prediction = "SUSPICIOUS"
            confidence = min(85, 50 + combined_score * 0.5)
        elif combined_score >= 20:
            prediction = "LIKELY REAL"
            confidence = min(80, 70 + (30 - combined_score))
        else:
            prediction = "REAL"
            confidence = min(99, 100 - combined_score)

        confidence = round(confidence, 1)

        # ── Generate Grad-CAM visualization ───────────────────────────────
        focus_regions = self._compute_focus_regions(forensics_result)
        gradcam_heatmap = self._generate_simulated_gradcam(img, focus_regions)
        gradcam_overlay = apply_heatmap_overlay(
            img, (gradcam_heatmap * 255).astype(np.float32),
            alpha=0.4, colormap=cv2.COLORMAP_JET
        )
        gradcam_b64 = cv2_to_base64(gradcam_overlay)

        # ── Build reasoning ───────────────────────────────────────────────
        reasoning = []

        # Biological signals first
        if forensics_result:
            if boundary_suspicious:
                reasoning.append("Blending artifacts at face boundaries (chin/hairline)")
            if eye_suspicious:
                reasoning.append("Mismatched eye reflections")

        # Primary detection signals
        if ela_score >= 25:
            reasoning.append(f"Compression inconsistencies detected (ELA score: {ela_score})")
        if dct_score >= 25:
            reasoning.append(f"Frequency domain anomalies found (DCT score: {dct_score})")

        # Secondary signals
        if forensics_result:
            if symmetry_suspicious:
                reasoning.append("Unusually symmetric face (secondary indicator)")
            if mouth_suspicious and forensics_result.get("mouth", {}).get("has_boundary_discontinuity", False):
                reasoning.append("Mouth rendering anomalies with boundary discontinuity")

        if flagged_count >= 2 and prediction not in ("REAL", "LIKELY REAL"):
            reasoning.append(f"Multiple detection layers agree ({flagged_count}/3 flagged)")

        if ela_dct_flagged >= 2 and prediction not in ("REAL", "LIKELY REAL"):
            reasoning.append("Both pixel-level (ELA) and frequency (DCT) analysis flag anomalies")

        if not reasoning:
            if prediction == "REAL":
                reasoning.append("All forensic checks passed — no significant anomalies detected")
            else:
                reasoning.append("Combined analysis suggests potential AI generation")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "combined_score": combined_score,
            "gradcam_b64": gradcam_b64,
            "model_loaded": False,
            "model_note": "Running in demo mode — using multi-layer forensic aggregation. For higher accuracy, add a trained model to backend/models/",
            "reasoning": reasoning,
            "fp_suppression_applied": False,
            "component_scores": {
                "ela": round(ela_score, 1),
                "ela_effective": round(ela_score, 1),
                "dct": round(dct_score, 1),
                "dct_effective": round(dct_score, 1),
                "forensics": round(forensics_score, 1),
            },
        }

    def _real_inference(self, img: np.ndarray, img_bytes: bytes) -> Dict[str, Any]:
        """Real model inference path (used when a model is loaded)."""
        # Placeholder for actual model inference
        # This would be implemented based on the specific model architecture
        return {
            "prediction": "UNKNOWN",
            "confidence": 0,
            "gradcam_b64": "",
            "model_loaded": True,
            "reasoning": ["Model inference not yet implemented for this architecture"],
        }
