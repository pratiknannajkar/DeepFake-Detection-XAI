"""
DeepShield AI — DCT Frequency Domain Analysis
Detects GAN/Diffusion model fingerprints in the frequency spectrum.

How it works:
1. Convert image to grayscale
2. Apply 2D Discrete Cosine Transform (DCT)
3. Analyze the frequency distribution for upsampling artifacts
4. GAN-generated images show periodic grid patterns in high frequencies
"""

import numpy as np
import cv2
from scipy.fft import dctn
from typing import Dict, Any

from backend.utils.image_utils import (
    cv2_to_base64,
    apply_heatmap_overlay,
    load_image_from_bytes,
    to_grayscale,
)


class DCTAnalyzer:
    """Frequency domain analysis using DCT to detect generative model artifacts."""

    def __init__(self, block_size: int = 8):
        """
        Args:
            block_size: Size of blocks for block-wise DCT (standard is 8x8, like JPEG)
        """
        self.block_size = block_size

    def compute_full_dct(self, img_gray: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to the full image and return log-scaled spectral map."""
        img_float = img_gray.astype(np.float64)
        dct_result = dctn(img_float, type=2, norm="ortho")

        # Log scale for visualization (avoids huge dynamic range)
        epsilon = 1e-12
        spectral_map = np.log(np.abs(dct_result) + epsilon)

        return spectral_map

    def compute_block_dct(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Apply DCT in 8x8 blocks (JPEG-style) and compute the average
        energy per coefficient position across all blocks.
        This is more sensitive to periodic upsampling artifacts.
        """
        h, w = img_gray.shape
        bs = self.block_size

        # Trim image to be divisible by block_size
        h_trim = (h // bs) * bs
        w_trim = (w // bs) * bs
        img_trimmed = img_gray[:h_trim, :w_trim].astype(np.float64)

        # Accumulate DCT coefficients across blocks
        energy_map = np.zeros((bs, bs), dtype=np.float64)
        block_count = 0

        for y in range(0, h_trim, bs):
            for x in range(0, w_trim, bs):
                block = img_trimmed[y : y + bs, x : x + bs]
                dct_block = cv2.dct(block)
                energy_map += np.abs(dct_block)
                block_count += 1

        if block_count > 0:
            energy_map /= block_count

        return energy_map

    def analyze_frequency_distribution(self, spectral_map: np.ndarray) -> Dict[str, float]:
        """
        Analyze the frequency distribution of the spectral map.
        Returns metrics about low/mid/high frequency energy distribution.
        """
        h, w = spectral_map.shape

        # Divide into frequency bands
        low_freq = spectral_map[: h // 4, : w // 4]
        mid_freq_mask = np.zeros_like(spectral_map, dtype=bool)
        mid_freq_mask[h // 4 : h // 2, w // 4 : w // 2] = True
        mid_freq = spectral_map[mid_freq_mask]

        high_freq_mask = np.zeros_like(spectral_map, dtype=bool)
        high_freq_mask[h // 2 :, w // 2 :] = True
        high_freq = spectral_map[high_freq_mask]

        # Compute energy ratios
        total_energy = float(np.sum(np.abs(spectral_map)))
        if total_energy == 0:
            total_energy = 1.0

        low_energy = float(np.sum(np.abs(low_freq)))
        mid_energy = float(np.sum(np.abs(mid_freq)))
        high_energy = float(np.sum(np.abs(high_freq)))

        return {
            "low_freq_ratio": round(low_energy / total_energy, 4),
            "mid_freq_ratio": round(mid_energy / total_energy, 4),
            "high_freq_ratio": round(high_energy / total_energy, 4),
            "high_to_low_ratio": round(
                high_energy / max(low_energy, 1e-12), 4
            ),
        }

    def detect_periodic_artifacts(self, energy_map: np.ndarray) -> Dict[str, Any]:
        """
        Detect periodic patterns in the block DCT energy map.
        GAN upsampling layers create characteristic "grid" patterns.
        """
        bs = self.block_size

        # Exclude DC component (top-left)
        ac_coefficients = energy_map.copy()
        ac_coefficients[0, 0] = 0

        # Check for unusual peaks in AC coefficients
        mean_ac = float(np.mean(ac_coefficients))
        std_ac = float(np.std(ac_coefficients))
        max_ac = float(np.max(ac_coefficients))

        # Count coefficients that are significantly above the mean
        threshold = mean_ac + 2 * std_ac
        peak_count = int(np.sum(ac_coefficients > threshold))

        # Periodicity check: look for repeating patterns
        # GAN artifacts often show energy at specific periodic positions
        has_periodic = False
        if std_ac > 0:
            normalized = (ac_coefficients - mean_ac) / std_ac
            # Check diagonals and specific positions for GAN fingerprints
            diagonal_energy = float(np.mean([np.abs(normalized[i, i]) for i in range(1, bs)]))
            anti_diag_energy = float(
                np.mean([np.abs(normalized[i, bs - 1 - i]) for i in range(bs - 1)])
            )
            has_periodic = diagonal_energy > 2.0 or anti_diag_energy > 2.0

        return {
            "mean_ac_energy": round(mean_ac, 4),
            "std_ac_energy": round(std_ac, 4),
            "max_ac_energy": round(max_ac, 4),
            "peak_count": peak_count,
            "has_periodic_artifacts": has_periodic,
        }

    def analyze(self, img_bytes: bytes) -> Dict[str, Any]:
        """
        Full DCT frequency analysis pipeline.

        Returns:
            dict with:
              - spectral_map_b64: base64 full DCT spectral map visualization
              - spectral_overlay_b64: base64 spectral map overlaid on original
              - block_energy_map: 8x8 average block DCT energy (as list)
              - frequency_distribution: low/mid/high frequency ratios
              - periodic_artifacts: periodic pattern detection results
              - overall_score: 0-100 suspicion score
              - verdict: textual interpretation
        """
        img_cv2 = load_image_from_bytes(img_bytes)
        img_gray = to_grayscale(img_cv2)

        # Resize for consistent analysis
        target_size = 512
        img_gray_resized = cv2.resize(img_gray, (target_size, target_size))

        # Full DCT spectral map
        spectral_map = self.compute_full_dct(img_gray_resized)

        # Block-wise DCT energy
        energy_map = self.compute_block_dct(img_gray_resized)

        # Analyze frequency distribution
        freq_dist = self.analyze_frequency_distribution(spectral_map)

        # Detect periodic artifacts
        periodic = self.detect_periodic_artifacts(energy_map)

        # Visualize spectral map
        spectral_vis = cv2.normalize(spectral_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        spectral_color = cv2.applyColorMap(spectral_vis, cv2.COLORMAP_INFERNO)
        spectral_map_b64 = cv2_to_base64(spectral_color)

        # Create overlay
        spectral_resized = cv2.resize(spectral_vis, (img_cv2.shape[1], img_cv2.shape[0]))
        overlay = apply_heatmap_overlay(
            img_cv2, spectral_resized.astype(np.float32), alpha=0.35, colormap=cv2.COLORMAP_INFERNO
        )
        spectral_overlay_b64 = cv2_to_base64(overlay)

        # Compute overall score
        score = 0.0

        # High-frequency energy ratio (GANs and diffusion models)
        hf_ratio = freq_dist["high_freq_ratio"]
        if hf_ratio > 0.12:
            score += 28
        elif hf_ratio > 0.06:
            score += 18
        elif hf_ratio > 0.03:
            score += 8

        # High-to-low ratio anomaly
        hl_ratio = freq_dist["high_to_low_ratio"]
        if hl_ratio > 0.4:
            score += 22
        elif hl_ratio > 0.15:
            score += 12
        elif hl_ratio > 0.08:
            score += 5

        # Mid-frequency anomaly (diffusion models often boost mid-frequencies)
        mf_ratio = freq_dist["mid_freq_ratio"]
        if mf_ratio > 0.25:
            score += 15
        elif mf_ratio > 0.15:
            score += 8

        # Periodic artifacts
        if periodic["has_periodic_artifacts"]:
            score += 30

        # Peak count in block DCT
        if periodic["peak_count"] > 8:
            score += 18
        elif periodic["peak_count"] > 4:
            score += 10
        elif periodic["peak_count"] > 2:
            score += 5

        # AC energy anomaly
        if periodic["std_ac_energy"] > 80:
            score += 12
        elif periodic["std_ac_energy"] > 50:
            score += 6

        score = min(100, round(score, 1))

        # Verdict
        if score >= 70:
            verdict = "HIGH RISK — Strong frequency-domain anomalies detected. The image shows patterns consistent with GAN or diffusion model generation (upsampling artifacts, periodic grid patterns)."
        elif score >= 40:
            verdict = "MODERATE RISK — Some unusual frequency patterns found. Could indicate partial AI generation or heavy post-processing."
        elif score >= 20:
            verdict = "LOW RISK — Minor frequency anomalies present, likely from normal image processing."
        else:
            verdict = "MINIMAL RISK — Frequency distribution appears natural with no significant generative model fingerprints."

        return {
            "spectral_map_b64": spectral_map_b64,
            "spectral_overlay_b64": spectral_overlay_b64,
            "block_energy_map": energy_map.tolist(),
            "frequency_distribution": freq_dist,
            "periodic_artifacts": periodic,
            "overall_score": score,
            "verdict": verdict,
        }
