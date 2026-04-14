"""
DeepShield AI — Error Level Analysis (ELA)
Detects compression inconsistencies that indicate image manipulation.

Key insight: scoring uses RAW pixel differences (before brightness enhancement)
to avoid false positives. The enhanced image is only for visualization.

Advanced features:
  - Face-region vs background differential (deepfakes show uneven ELA)
  - Multi-quality sweep (tests multiple JPEG quality levels)
  - Noise consistency analysis
"""

import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, Tuple

from backend.utils.image_utils import (
    pil_to_base64,
    cv2_to_base64,
    apply_heatmap_overlay,
    load_pil_from_bytes,
    load_image_from_bytes,
)
from backend.utils.face_extractor import FaceExtractor


class ELAAnalyzer:
    """Error Level Analysis for detecting image tampering/generation."""

    def __init__(self, quality: int = 90, scale: float = 15.0):
        self.quality = quality
        self.scale = scale
        self.face_extractor = FaceExtractor()

    def _compute_raw_ela(self, img_pil: Image.Image, quality: int) -> np.ndarray:
        """
        Compute RAW ELA difference (no brightness enhancement).
        Returns float32 array with actual pixel-level differences (0-255 range).
        """
        buffer = io.BytesIO()
        img_pil.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(img_pil, resaved)
        return np.array(diff, dtype=np.float32)

    def _compute_enhanced_ela(self, raw_diff: np.ndarray) -> Image.Image:
        """Create brightness-enhanced ELA for visualization only."""
        diff_img = Image.fromarray(raw_diff.astype(np.uint8))
        extrema = diff_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        brightness_scale = min(255.0 / max_diff * self.scale, 100.0)
        return ImageEnhance.Brightness(diff_img).enhance(brightness_scale)

    def _face_vs_background_analysis(
        self, raw_gray: np.ndarray, img_cv2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare ELA levels in the face region vs background.
        In deepfakes, the face region typically has DIFFERENT error levels
        than the background because it was generated/pasted separately.
        In real photos, ELA is relatively uniform.
        """
        faces = self.face_extractor.detect_faces(img_cv2)

        if not faces:
            return {
                "face_detected": False,
                "differential": 0,
                "face_mean": 0,
                "bg_mean": 0,
                "detail": "No face detected for differential analysis",
            }

        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Ensure bounds are valid
        img_h, img_w = raw_gray.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        # Extract face and background regions
        face_mask = np.zeros(raw_gray.shape[:2], dtype=bool)
        face_mask[y:y2, x:x2] = True

        face_ela = raw_gray[face_mask]
        bg_ela = raw_gray[~face_mask]

        if face_ela.size == 0 or bg_ela.size == 0:
            return {
                "face_detected": True,
                "differential": 0,
                "face_mean": 0,
                "bg_mean": 0,
                "detail": "Could not compute differential",
            }

        face_mean = float(np.mean(face_ela))
        bg_mean = float(np.mean(bg_ela))
        face_std = float(np.std(face_ela))
        bg_std = float(np.std(bg_ela))

        # The differential: how different is the face ELA from background
        differential = abs(face_mean - bg_mean)
        # Relative differential (normalized by overall mean)
        overall_mean = float(np.mean(raw_gray))
        relative_diff = differential / max(overall_mean, 0.1)

        detail = ""
        if relative_diff > 1.5:
            detail = "STRONG mismatch — face region has very different compression signature than background (classic deepfake indicator)"
        elif relative_diff > 0.8:
            detail = "MODERATE mismatch — face and background show different error levels"
        elif relative_diff > 0.4:
            detail = "SLIGHT mismatch — minor ELA difference between face and background"
        else:
            detail = "CONSISTENT — face and background have similar ELA profiles (expected for real photos)"

        return {
            "face_detected": True,
            "differential": round(differential, 3),
            "relative_differential": round(relative_diff, 3),
            "face_mean": round(face_mean, 3),
            "face_std": round(face_std, 3),
            "bg_mean": round(bg_mean, 3),
            "bg_std": round(bg_std, 3),
            "detail": detail,
        }

    def _multi_quality_analysis(self, img_pil: Image.Image) -> Dict[str, Any]:
        """
        Test ELA at multiple JPEG quality levels.
        Manipulated images show inconsistent error patterns across qualities.
        Real images degrade predictably.
        """
        qualities = [75, 85, 95]
        means = []

        for q in qualities:
            raw = self._compute_raw_ela(img_pil, q)
            gray = np.mean(raw, axis=2) if len(raw.shape) == 3 else raw
            means.append(float(np.mean(gray)))

        # In real images, error decreases smoothly as quality increases
        # In fakes, the pattern is often non-monotonic or has jumps
        diffs = [means[i] - means[i + 1] for i in range(len(means) - 1)]
        is_monotonic = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)

        # Variation coefficient across qualities
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        variation = std_of_means / max(mean_of_means, 0.01)

        return {
            "quality_levels": qualities,
            "mean_errors": [round(m, 3) for m in means],
            "is_monotonic": is_monotonic,
            "variation_coefficient": round(float(variation), 4),
        }

    def _noise_consistency_check(self, img_cv2: np.ndarray) -> Dict[str, Any]:
        """
        Real photos have uniform sensor noise. AI-generated images
        often have inconsistent noise patterns across different regions.
        """
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2

        # Extract noise using high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred).astype(np.float32)

        h, w = noise.shape
        # Divide into 4x4 grid and measure noise variance in each cell
        grid_rows, grid_cols = 4, 4
        cell_h, cell_w = h // grid_rows, w // grid_cols
        variances = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = noise[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                variances.append(float(np.var(cell)))

        mean_var = float(np.mean(variances))
        std_var = float(np.std(variances))

        # Coefficient of variation of noise variances
        # Real photos: low (uniform noise) → CV < 0.3
        # AI images: high (inconsistent noise) → CV > 0.5
        noise_cv = std_var / max(mean_var, 0.01)

        if noise_cv > 0.8:
            detail = "Highly inconsistent noise pattern — strong indicator of synthetic content"
        elif noise_cv > 0.5:
            detail = "Moderately inconsistent noise — possible AI generation"
        elif noise_cv > 0.3:
            detail = "Slightly uneven noise — could be normal or mildly edited"
        else:
            detail = "Uniform noise pattern — consistent with real camera sensor"

        return {
            "noise_cv": round(noise_cv, 4),
            "mean_noise_variance": round(mean_var, 4),
            "noise_std": round(std_var, 4),
            "detail": detail,
        }

    def analyze(self, img_bytes: bytes) -> Dict[str, Any]:
        """
        Full ELA analysis pipeline with advanced features.
        Scoring is based on RAW pixel differences, NOT the enhanced visualization.
        """
        img_pil = load_pil_from_bytes(img_bytes)
        img_cv2 = load_image_from_bytes(img_bytes)

        # ── Step 1: Raw ELA (for scoring) ─────────────────────────────────
        raw_diff = self._compute_raw_ela(img_pil, self.quality)
        # Convert to grayscale for scoring
        raw_gray = np.mean(raw_diff, axis=2) if len(raw_diff.shape) == 3 else raw_diff

        raw_mean = float(np.mean(raw_gray))
        raw_max = float(np.max(raw_gray))
        raw_std = float(np.std(raw_gray))
        raw_p95 = float(np.percentile(raw_gray, 95))

        # ── Step 2: Enhanced ELA (for visualization) ──────────────────────
        ela_enhanced = self._compute_enhanced_ela(raw_diff)
        enhanced_gray = np.array(ela_enhanced.convert("L"), dtype=np.float32)

        # ── Step 3: Face vs Background differential ───────────────────────
        face_bg = self._face_vs_background_analysis(raw_gray, img_cv2)

        # ── Step 4: Multi-quality analysis ────────────────────────────────
        multi_q = self._multi_quality_analysis(img_pil)

        # ── Step 5: Noise consistency ─────────────────────────────────────
        noise = self._noise_consistency_check(img_cv2)

        # ── SCORING (from RAW differences, not enhanced) ──────────────────
        # Typical raw ELA values:
        #   Real JPEG (q90): mean ~2-5, p95 ~8-15
        #   Manipulated:     mean ~5-15, p95 ~20-50+
        #   AI-generated:    mean ~3-10, p95 ~15-40
        score = 0.0

        # Signal 1: Raw mean error (real images ~2-5, AI ~3-10, spliced ~8-15+)
        if raw_mean > 10:
            score += 22
        elif raw_mean > 6:
            score += 15
        elif raw_mean > 3.5:
            score += 8
        elif raw_mean > 2.5:
            score += 4

        # Signal 2: Raw 95th percentile (real ~8-12, AI ~12-35, spliced ~25-50+)
        if raw_p95 > 35:
            score += 20
        elif raw_p95 > 20:
            score += 14
        elif raw_p95 > 12:
            score += 8
        elif raw_p95 > 8:
            score += 3

        # Signal 3: Face-vs-background differential (THE key deepfake signal)
        rel_diff = face_bg.get("relative_differential", 0)
        if rel_diff > 1.2:
            score += 30  # Very strong indicator
        elif rel_diff > 0.6:
            score += 22
        elif rel_diff > 0.3:
            score += 12
        elif rel_diff > 0.15:
            score += 5

        # Signal 4: Multi-quality consistency
        if not multi_q["is_monotonic"]:
            score += 12
        if multi_q["variation_coefficient"] > 0.4:
            score += 10
        elif multi_q["variation_coefficient"] > 0.2:
            score += 5

        # Signal 5: Noise inconsistency OR suspicious uniformity
        # Real camera photos: moderate noise variation (CV 0.2-0.5)
        # AI-generated images:  extremely uniform/low noise (CV < 0.1) OR highly inconsistent
        # Heavily re-compressed social media: moderate-high noise (CV 0.3-0.6)
        noise_cv_val = noise["noise_cv"]
        if noise_cv_val > 0.7:
            score += 20
        elif noise_cv_val > 0.4:
            score += 12
        elif noise_cv_val > 0.25:
            score += 6
        elif noise_cv_val > 0.15:
            score += 3
        # AI synthesis signal: suspiciously perfect/uniform noise (too clean for a camera)
        if noise_cv_val < 0.08 and raw_mean > 2.0:
            score += 15  # Unnaturally uniform — strong AI generation marker

        score = min(100, round(score, 1))

        # ── FALSE POSITIVE SUPPRESSION ────────────────────────────────────
        # Only suppress when ALL of these are true:
        #   1. Score is in moderate range (not very high)
        #   2. Noise is consistent (suggesting real camera, not AI synthesis)
        #   3. Face-vs-background differential is low (uniform image)
        #   4. Raw mean error is low-moderate (not extremely anomalous)
        noise_cv = noise.get("noise_cv", 1.0)
        compression_fp_suppressed = False
        rel_diff_val = face_bg.get("relative_differential", 0)

        # Only suppress if score is moderate AND noise looks like typical re-compressed JPEG
        # Do NOT suppress if noise_cv is very low (< 0.08) — that's an AI synthesis signal
        is_likely_recompressed = (
            noise_cv >= 0.10  # Not suspiciously uniform (not AI-smooth)
            and noise_cv < 0.40  # Consistent noise (not wildly inconsistent)
            and rel_diff_val < 0.25  # No face-bg differential
            and raw_mean < 8.0  # Low-moderate raw error
            and score < 55  # Only suppress moderate scores, not high ones
        )

        if is_likely_recompressed:
            score = max(0, round(score * 0.50))  # Suppress by 50% — model is primary now
            compression_fp_suppressed = True

        # ── Region scores ─────────────────────────────────────────────────
        regions = self._compute_region_scores(raw_gray)
        region_means = [r["mean_error"] for r in regions]
        region_variance = float(np.var(region_means))

        # ── Visualizations ────────────────────────────────────────────────
        ela_heatmap_b64 = pil_to_base64(ela_enhanced)
        overlay = apply_heatmap_overlay(img_cv2, enhanced_gray, alpha=0.4)
        ela_overlay_b64 = cv2_to_base64(overlay)

        # ── Verdict ───────────────────────────────────────────────────────
        if compression_fp_suppressed:
            verdict_suffix = " Note: Score was reduced due to consistent noise patterns suggesting normal JPEG compression."
        else:
            verdict_suffix = ""

        if score >= 60:
            verdict = "HIGH RISK — Significant compression inconsistencies and face-background mismatch detected. Strong evidence of manipulation or AI generation." + verdict_suffix
        elif score >= 35:
            verdict = "MODERATE RISK — Some compression anomalies found. The image may have been partially edited or AI-generated." + verdict_suffix
        elif score >= 18:
            verdict = "LOW RISK — Minor compression artifacts present, likely from normal JPEG processing." + verdict_suffix
        else:
            verdict = "MINIMAL RISK — Error levels are consistent across the image, suggesting no significant manipulation." + verdict_suffix

        return {
            "ela_heatmap_b64": ela_heatmap_b64,
            "ela_overlay_b64": ela_overlay_b64,
            "overall_score": score,
            "max_error": round(raw_max, 3),
            "mean_error": round(raw_mean, 3),
            "error_std": round(raw_std, 3),
            "p95_error": round(raw_p95, 3),
            "region_variance": round(region_variance, 3),
            "face_vs_background": face_bg,
            "multi_quality": multi_q,
            "noise_consistency": noise,
            "regions": regions,
            "verdict": verdict,
        }

    def _compute_region_scores(self, heatmap: np.ndarray, grid_rows: int = 4, grid_cols: int = 4) -> list:
        h, w = heatmap.shape[:2]
        cell_h, cell_w = h // grid_rows, w // grid_cols
        regions = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                cell = heatmap[y1:y2, x1:x2]
                regions.append({
                    "row": r, "col": c,
                    "mean_error": round(float(np.mean(cell)), 3),
                    "max_error": round(float(np.max(cell)), 3),
                })
        return regions
