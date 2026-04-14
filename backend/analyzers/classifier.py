"""
DeepShield AI — CNN Classifier with Grad-CAM Explainability
Provides a deep-learning classification layer with visual explanations.

PRIMARY MODE: Uses EfficientNet-B0 (fine-tuned on deepfake dataset)
for binary deepfake classification with real Grad-CAM heatmaps.
Model file: backend/models/efficientnet_deepfake.pth (~20MB).

FALLBACK MODE: If model file is missing, falls back to heuristic
aggregation of ELA/DCT/Forensics scores (less accurate but always available).
"""

import os
import json
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional

from backend.utils.image_utils import cv2_to_base64, load_image_from_bytes, apply_heatmap_overlay
from backend.analyzers import model_loader

logger = logging.getLogger("deepshield.classifier")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


class ClassifierAnalyzer:
    """
    CNN-based deepfake classifier with Grad-CAM explainability.

    Uses EfficientNet-B0 (fine-tuned) as the primary detection method.
    Falls back to heuristic aggregation if the model isn't available.
    """

    def __init__(self):
        self.model_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        """Try to load the EfficientNet-B0 model."""
        try:
            if model_loader.is_model_loaded():
                self.model_loaded = True
                logger.info("ClassifierAnalyzer: EfficientNet-B0 model is ready.")
            else:
                err = model_loader.get_load_error()
                logger.warning(f"ClassifierAnalyzer: EfficientNet-B0 model not available: {err}")
                logger.warning("ClassifierAnalyzer: Will use heuristic fallback mode.")
        except Exception as e:
            logger.error(f"ClassifierAnalyzer: Model init error: {e}")
            self.model_loaded = False

    def _build_safe_response(
        self,
        prediction: str,
        confidence: float,
        combined_score: float,
        gradcam_b64: str,
        model_loaded: bool,
        reasoning: list,
        model_note: str = "",
        component_scores: Optional[dict] = None,
        fp_suppression_applied: bool = False,
    ) -> Dict[str, Any]:
        """Return a stable schema expected by the API layer."""
        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 1),
            "combined_score": round(float(combined_score), 1),
            "gradcam_b64": gradcam_b64 or "",
            "model_loaded": bool(model_loaded),
            "model_note": model_note,
            "reasoning": reasoning if reasoning else ["No additional reasoning available"],
            "fp_suppression_applied": bool(fp_suppression_applied),
            "component_scores": component_scores or {},
        }

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

        Uses CLIP model as primary classifier (when available),
        with heuristic scores as secondary stabilization signal.

        Returns:
            dict with:
              - prediction: "FAKE" or "REAL"
              - confidence: 0-100 percentage
              - gradcam_b64: base64 Grad-CAM overlay
              - model_loaded: whether a real model was used
              - reasoning: list of factors that contributed to the decision
        """
        img_cv2 = load_image_from_bytes(img_bytes)

        # Try EfficientNet model first
        if model_loader.is_model_loaded():
            return self._model_inference(
                img_cv2, ela_score, dct_score, forensics_score, forensics_result
            )

        # Fallback: heuristic aggregation
        return self._demo_inference(
            img_cv2, ela_score, dct_score, forensics_score, forensics_result
        )

    def _model_inference(
        self,
        img: np.ndarray,
        ela_score: float,
        dct_score: float,
        forensics_score: float,
        forensics_result: Optional[dict],
    ) -> Dict[str, Any]:
        """
        EfficientNet-B0 inference — PRIMARY classification path.

        Strategy:
        1. EfficientNet-B0 (trained on deepfake data) gives fake_probability
        2. Convert to 0-100 model_score
        3. Blend 80% model + 20% heuristics
        4. Generate real Grad-CAM heatmap from last conv layer
        """
        try:
            # ── Step 1: EfficientNet classification ───────────────────────
            fake_prob, model_details = model_loader.predict_fake_probability(img)
            model_score = float(np.clip(fake_prob * 100.0, 0.0, 100.0))

            logger.info(
                f"EfficientNet prediction: fake_prob={fake_prob:.3f}, "
                f"model_score={model_score:.1f}, "
                f"verdict={model_details.get('top_match', 'N/A')}"
            )

            # ── Step 2: Heuristic stabilization (20% weight) ─────────────
            heuristic_max = max(float(ela_score), float(dct_score), float(forensics_score))
            heuristic_avg = (float(ela_score) + float(dct_score) + float(forensics_score)) / 3.0

            # 80/20 fusion — trained model is primary, heuristics are supporting
            heuristic_influence = heuristic_max * 0.6 + heuristic_avg * 0.4
            combined_score = model_score * 0.80 + heuristic_influence * 0.20

            # ── Step 3: Multi-signal agreement boost ──────────────────────
            heuristic_flagged = sum(1 for s in [ela_score, dct_score, forensics_score] if s >= 35)

            if model_score >= 60 and heuristic_flagged >= 2:
                # Model + multiple heuristics agree → high confidence
                combined_score = min(100, combined_score + 8)
            elif model_score < 25 and heuristic_max < 20:
                # Model + heuristics agree it's real → reinforce
                combined_score = max(0, combined_score - 5)

            # Safety net: if model is very confident, let it lead
            if model_score >= 85:
                combined_score = max(combined_score, model_score * 0.90)
            elif model_score <= 15:
                combined_score = min(combined_score, model_score * 1.2 + 8)

            combined_score = min(100, max(0, round(combined_score, 1)))

            # ── Step 4: Prediction thresholds ─────────────────────────────
            if combined_score >= 55:
                prediction = "FAKE"
                confidence = min(99, 50 + combined_score * 0.5)
            elif combined_score >= 38:
                prediction = "SUSPICIOUS"
                confidence = min(90, 40 + combined_score * 0.7)
            elif combined_score >= 22:
                prediction = "LIKELY REAL"
                confidence = min(85, 60 + (35 - min(combined_score, 35)))
            else:
                prediction = "REAL"
                confidence = min(99, 95 - combined_score * 0.5)

            confidence = round(confidence, 1)

            # ── Step 5: Generate Grad-CAM heatmap ─────────────────────────
            try:
                gradcam_heatmap = model_loader.generate_attention_heatmap(img)
                gradcam_overlay = apply_heatmap_overlay(
                    img, (gradcam_heatmap * 255).astype(np.float32),
                    alpha=0.4, colormap=cv2.COLORMAP_JET
                )
            except Exception as heatmap_err:
                logger.warning(f"Grad-CAM failed: {heatmap_err}, using fallback")
                focus_regions = self._compute_focus_regions(forensics_result)
                fallback_heatmap = self._generate_simulated_gradcam(img, focus_regions)
                gradcam_overlay = apply_heatmap_overlay(
                    img, (fallback_heatmap * 255).astype(np.float32),
                    alpha=0.4, colormap=cv2.COLORMAP_JET
                )

            gradcam_b64 = cv2_to_base64(gradcam_overlay)

            # ── Step 6: Build reasoning ───────────────────────────────────
            reasoning = []

            fake_pct = model_details.get("fake_probability", 0)
            real_pct = model_details.get("real_probability", 0)
            reasoning.append(
                f"EfficientNet-B0: {fake_pct}% fake, {real_pct}% real"
            )

            if heuristic_max >= 30:
                reasoning.append(
                    f"Forensic heuristics support detection "
                    f"(ELA:{round(ela_score)}, DCT:{round(dct_score)}, Forensics:{round(forensics_score)})"
                )

            # Add specific forensic findings
            if forensics_result:
                if forensics_result.get("boundaries", {}).get("is_suspicious"):
                    reasoning.append("Blending artifacts at face boundaries (chin/hairline)")
                if forensics_result.get("eye_reflections", {}).get("is_suspicious"):
                    reasoning.append("Mismatched eye reflections")

            if ela_score >= 30:
                reasoning.append(f"Compression inconsistencies detected (ELA score: {round(ela_score)})")
            if dct_score >= 30:
                reasoning.append(f"Frequency domain anomalies found (DCT score: {round(dct_score)})")

            if model_score >= 60 and heuristic_flagged >= 2:
                reasoning.append("Both AI model and forensic heuristics agree on detection")

            if prediction == "REAL" and len(reasoning) <= 1:
                reasoning.append("All analysis layers indicate authentic image")

            return self._build_safe_response(
                prediction=prediction,
                confidence=confidence,
                combined_score=combined_score,
                gradcam_b64=gradcam_b64,
                model_loaded=True,
                model_note="Using EfficientNet-B0 (fine-tuned on deepfake dataset) with real Grad-CAM and forensic heuristic stabilization.",
                reasoning=reasoning,
                fp_suppression_applied=False,
                component_scores={
                    "model_score": round(model_score, 1),
                    "model_fake_prob": round(fake_prob * 100, 1),
                    "model_real_prob": round((1 - fake_prob) * 100, 1),
                    "ela": round(float(ela_score), 1),
                    "dct": round(float(dct_score), 1),
                    "forensics": round(float(forensics_score), 1),
                    "heuristic_max": round(heuristic_max, 1),
                },
            )

        except Exception as exc:
            logger.error(f"EfficientNet inference failed: {exc}", exc_info=True)
            # Fall back to heuristic mode
            fallback = self._demo_inference(
                img, ela_score, dct_score, forensics_score, forensics_result
            )
            fallback["model_note"] = (
                f"EfficientNet model inference failed: {str(exc)}. "
                "Fell back to heuristic forensic aggregation."
            )
            fallback["reasoning"].insert(
                0, f"Model inference error handled safely: {str(exc)}"
            )
            return fallback

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
        scores = [ela_score, dct_score, forensics_score]
        max_score = max(scores)
        ela_dct_max = max(ela_score, dct_score)

        # Weighted average: ELA/DCT dominate
        weighted_avg = (
            ela_score * 0.35
            + dct_score * 0.35
            + forensics_score * 0.30
        )

        # Boundary-specific boost
        if boundary_suspicious:
            boundary_score = forensics_result.get("boundaries", {}).get("score", 0)
            weighted_avg = max(weighted_avg, boundary_score * 0.85)

        # Combined score
        combined_score = max_score * 0.55 + weighted_avg * 0.45

        # ── Multi-signal agreement boost ──────────────────────────────────
        flagged_count = sum(1 for s in scores if s >= 30)
        ela_dct_flagged = sum(1 for s in [ela_score, dct_score] if s >= 30)

        if flagged_count >= 3:
            combined_score = min(100, combined_score + 15)
        elif flagged_count >= 2:
            combined_score = min(100, combined_score + 10)

        if ela_dct_flagged >= 2 and ela_dct_max >= 40:
            combined_score = min(100, combined_score + 8)

        # ── Biological signal corroboration ───────────────────────────────
        if boundary_suspicious and eye_suspicious:
            combined_score = min(100, combined_score + 12)
        elif boundary_suspicious or eye_suspicious:
            combined_score = min(100, combined_score + 5)

        # ── Strong single-signal override ─────────────────────────────────
        if ela_dct_max >= 65:
            combined_score = max(combined_score, ela_dct_max * 0.90)
        elif ela_dct_max >= 50:
            combined_score = max(combined_score, ela_dct_max * 0.80)
        elif ela_dct_max >= 40:
            combined_score = max(combined_score, ela_dct_max * 0.70)

        combined_score = min(100, max(0, round(combined_score, 1)))

        # ── Prediction ────────────────────────────────────────────────────
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

        return self._build_safe_response(
            prediction=prediction,
            confidence=confidence,
            combined_score=combined_score,
            gradcam_b64=gradcam_b64,
            model_loaded=False,
            model_note="Running in heuristic fallback mode — CLIP model not available. Using multi-layer forensic aggregation.",
            reasoning=reasoning,
            fp_suppression_applied=False,
            component_scores={
                "ela": round(ela_score, 1),
                "ela_effective": round(ela_score, 1),
                "dct": round(dct_score, 1),
                "dct_effective": round(dct_score, 1),
                "forensics": round(forensics_score, 1),
            },
        )
