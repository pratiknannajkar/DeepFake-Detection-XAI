"""
DeepShield AI — Face Forensics Analyzer
Checks biometric and visual heuristics that AI-generated faces often get wrong.

Detection targets:
1. Facial symmetry (too-perfect = suspicious)
2. Eye reflection consistency
3. Boundary artifacts (hairline, chin edges)
4. Mouth/teeth rendering quality
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple

from backend.utils.image_utils import cv2_to_base64, load_image_from_bytes
from backend.utils.face_extractor import FaceExtractor


class FaceForensicsAnalyzer:
    """Heuristic face forensics analysis for deepfake detection."""

    def __init__(self):
        self.face_extractor = FaceExtractor()

    def _compute_symmetry_score(self, landmarks: dict, img: np.ndarray) -> Dict[str, Any]:
        """
        Measure facial symmetry. Real faces have natural asymmetry.
        Too-perfect symmetry indicates AI generation.

        RECALIBRATION: For high-resolution portraits (width > 512px),
        symmetry appears more precise due to pixel density, NOT because
        the face is AI-generated. Apply a damping factor to avoid
        false positives on high-res real portraits.
        """
        if not landmarks:
            return {"score": 50, "detail": "No landmarks available", "is_suspicious": False}

        h, w = img.shape[:2]
        all_lm = landmarks.get("all_landmarks", [])
        if len(all_lm) < 400:
            return {"score": 50, "detail": "Insufficient landmarks", "is_suspicious": False}

        # Find nose tip (landmark 1) as center axis
        nose_tip = all_lm[1]
        center_x = nose_tip["x"]

        # Compare left vs right landmark positions (mirrored around nose)
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])
        left_brow = landmarks.get("left_eyebrow", [])
        right_brow = landmarks.get("right_eyebrow", [])

        deviations = []

        # Compare eye positions
        if left_eye and right_eye:
            left_eye_center = np.mean([(p["x"], p["y"]) for p in left_eye], axis=0)
            right_eye_center = np.mean([(p["x"], p["y"]) for p in right_eye], axis=0)

            # Distance from center
            left_dist = abs(center_x - left_eye_center[0])
            right_dist = abs(right_eye_center[0] - center_x)

            if max(left_dist, right_dist) > 0:
                eye_symmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                deviations.append(eye_symmetry)

            # Y-level difference
            y_diff = abs(left_eye_center[1] - right_eye_center[1]) / h
            deviations.append(y_diff)

        # Compare eyebrow positions
        if left_brow and right_brow:
            left_brow_center = np.mean([(p["x"], p["y"]) for p in left_brow], axis=0)
            right_brow_center = np.mean([(p["x"], p["y"]) for p in right_brow], axis=0)

            left_dist = abs(center_x - left_brow_center[0])
            right_dist = abs(right_brow_center[0] - center_x)

            if max(left_dist, right_dist) > 0:
                brow_symmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                deviations.append(brow_symmetry)

        if not deviations:
            return {"score": 50, "detail": "Could not compute symmetry", "is_suspicious": False}

        avg_deviation = float(np.mean(deviations))

        # Low deviation = very symmetric = suspicious (AI-generated)
        # Normal faces have deviation ~0.05-0.15
        # AI faces often have deviation < 0.03
        if avg_deviation < 0.02:
            score = 90
            detail = "Unnaturally perfect symmetry detected — faces this symmetric are rare in nature"
            suspicious = True
        elif avg_deviation < 0.04:
            score = 72
            detail = "Very high symmetry — strong indicator of AI generation"
            suspicious = True
        elif avg_deviation < 0.07:
            score = 55
            detail = "Unusually symmetric face — could indicate AI generation"
            suspicious = True
        elif avg_deviation < 0.15:
            score = 25
            detail = "Normal facial asymmetry — consistent with real human faces"
            suspicious = False
        else:
            score = 15
            detail = "Natural asymmetry present — typical of real photographs"
            suspicious = False

        # HIGH-RES DAMPING: In high-resolution portraits, sub-pixel landmark
        # precision makes real faces appear more symmetric than they truly are.
        # Dampen the score for images wider than 512px to compensate.
        if w > 512:
            damping = min(0.7, 512.0 / w + 0.3)  # e.g. 1024px → 0.8, 2048px → 0.55
            score = round(score * damping)
            if score < 40:
                suspicious = False
                detail += " (dampened for high-resolution portrait)"

        return {
            "score": score,
            "avg_deviation": round(avg_deviation, 4),
            "detail": detail,
            "is_suspicious": suspicious,
        }

    def _check_eye_reflections(self, landmarks: dict, img: np.ndarray) -> Dict[str, Any]:
        """
        Compare light reflections in both eyes.
        Real photos have consistent reflections; AI often mismatches them.
        """
        if not landmarks:
            return {"score": 50, "detail": "No landmarks available", "is_suspicious": False}

        left_iris = landmarks.get("left_iris", [])
        right_iris = landmarks.get("right_iris", [])

        if not left_iris or not right_iris:
            return {"score": 50, "detail": "Iris landmarks not available", "is_suspicious": False}

        # Extract small patches around each iris center
        left_center = left_iris[0]
        right_center = right_iris[0]

        patch_size = 20
        h, w = img.shape[:2]

        def extract_eye_patch(center, size):
            x, y = center["x"], center["y"]
            x1 = max(0, x - size)
            y1 = max(0, y - size)
            x2 = min(w, x + size)
            y2 = min(h, y + size)
            return img[y1:y2, x1:x2]

        left_patch = extract_eye_patch(left_center, patch_size)
        right_patch = extract_eye_patch(right_center, patch_size)

        if left_patch.size == 0 or right_patch.size == 0:
            return {"score": 50, "detail": "Could not extract eye regions", "is_suspicious": False}

        # Analyze brightness patterns (reflections = bright spots)
        left_gray = cv2.cvtColor(left_patch, cv2.COLOR_BGR2GRAY) if len(left_patch.shape) == 3 else left_patch
        right_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY) if len(right_patch.shape) == 3 else right_patch

        # Resize to same dimensions for comparison
        target = (30, 30)
        left_resized = cv2.resize(left_gray, target).astype(np.float64)
        right_resized = cv2.resize(right_gray, target).astype(np.float64)

        # Flip right eye horizontally for comparison (mirrors should match)
        right_flipped = cv2.flip(right_resized, 1)

        # Compute similarity
        diff = np.abs(left_resized - right_flipped)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        # Check for brightness peak consistency (reflection spots)
        left_max = float(np.max(left_resized))
        right_max = float(np.max(right_resized))
        brightness_diff = abs(left_max - right_max)

        # Histogram comparison
        left_hist = cv2.calcHist([left_resized.astype(np.uint8)], [0], None, [32], [0, 256])
        right_hist = cv2.calcHist([right_flipped.astype(np.uint8)], [0], None, [32], [0, 256])
        hist_correlation = float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL))

        # Score: low correlation or high brightness diff = suspicious
        score = 0
        if hist_correlation < 0.5:
            score += 40
        elif hist_correlation < 0.7:
            score += 20

        if brightness_diff > 60:
            score += 30
        elif brightness_diff > 30:
            score += 15

        if mean_diff > 40:
            score += 20
        elif mean_diff > 25:
            score += 10

        score = min(100, score)

        if score >= 55:
            detail = "Eye reflections are inconsistent — AI models often fail to render matching light sources in both eyes"
            suspicious = True
        elif score >= 28:
            detail = "Minor reflection differences detected — could be lighting angle or AI artifact"
            suspicious = True
        else:
            detail = "Eye reflections appear consistent — matches expected physics of real lighting"
            suspicious = False

        return {
            "score": score,
            "hist_correlation": round(hist_correlation, 4),
            "brightness_diff": round(brightness_diff, 2),
            "mean_pixel_diff": round(mean_diff, 2),
            "detail": detail,
            "is_suspicious": suspicious,
        }

    def _check_boundary_artifacts(self, landmarks: dict, img: np.ndarray) -> Dict[str, Any]:
        """
        Check for blending artifacts at face boundaries (hairline, chin, ears).
        AI faces often have soft/blurry transitions at these edges.
        """
        if not landmarks:
            return {"score": 50, "detail": "No landmarks available", "is_suspicious": False}

        jaw = landmarks.get("jaw", [])
        forehead = landmarks.get("forehead_center", [])

        if len(jaw) < 10:
            return {"score": 50, "detail": "Insufficient jaw landmarks", "is_suspicious": False}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Compute gradient magnitude along jaw/chin boundary
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Sample gradient values along jaw line
        jaw_gradients = []
        for point in jaw:
            x, y = point["x"], point["y"]
            if 0 <= y < h and 0 <= x < w:
                # Sample a small neighborhood
                y1, y2 = max(0, y - 3), min(h, y + 4)
                x1, x2 = max(0, x - 3), min(w, x + 4)
                patch = gradient_mag[y1:y2, x1:x2]
                if patch.size > 0:
                    jaw_gradients.append(float(np.mean(patch)))

        if not jaw_gradients:
            return {"score": 50, "detail": "Could not sample boundaries", "is_suspicious": False}

        mean_gradient = float(np.mean(jaw_gradients))
        std_gradient = float(np.std(jaw_gradients))
        global_gradient_mean = float(np.mean(gradient_mag))

        # Very low gradient at boundaries = blurry blending (AI artifact)
        # Very inconsistent gradients = poor blending
        score = 0

        boundary_ratio = mean_gradient / max(global_gradient_mean, 1.0)

        if boundary_ratio < 0.3:
            score += 40
        elif boundary_ratio < 0.6:
            score += 20

        # High variance in boundary gradients = inconsistent blending
        if std_gradient > mean_gradient * 1.5:
            score += 25
        elif std_gradient > mean_gradient:
            score += 10

        # Check for "halo" effect — unusually bright/dark band at boundaries
        boundary_values = []
        for point in jaw[:len(jaw)//2]:  # Sample half the jaw
            x, y = point["x"], point["y"]
            if 5 <= y < h - 5 and 5 <= x < w - 5:
                inner = float(np.mean(gray[y - 5:y, x - 2:x + 3]))
                outer = float(np.mean(gray[y:y + 5, x - 2:x + 3]))
                boundary_values.append(abs(inner - outer))

        if boundary_values:
            avg_boundary_contrast = float(np.mean(boundary_values))
            if avg_boundary_contrast < 5:
                score += 20  # Suspiciously smooth boundary

        score = min(100, score)

        if score >= 50:
            detail = "Boundary artifacts detected — blurry or inconsistent edges at face boundaries suggest AI-generated blending"
            suspicious = True
        elif score >= 25:
            detail = "Some boundary irregularities found — could be compression or AI artifacts"
            suspicious = True
        else:
            detail = "Face boundaries appear natural with expected edge transitions"
            suspicious = False

        return {
            "score": score,
            "mean_boundary_gradient": round(mean_gradient, 2),
            "gradient_std": round(std_gradient, 2),
            "boundary_to_global_ratio": round(boundary_ratio, 4),
            "detail": detail,
            "is_suspicious": suspicious,
        }

    def _check_mouth_region(self, landmarks: dict, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mouth/teeth rendering quality with spatial threshold.

        SPATIAL THRESHOLD FIX: Do NOT flag 'Unnatural Teeth/Mouth' UNLESS
        the Grad-CAM attention map shows a sharp boundary discontinuity
        that contradicts the surrounding skin texture. This prevents false
        positives from lighting, motion blur, or JPEG compression around lips.
        """
        if not landmarks:
            return {"score": 50, "detail": "No landmarks available", "is_suspicious": False}

        mouth = landmarks.get("mouth", [])
        if len(mouth) < 10:
            return {"score": 50, "detail": "Insufficient mouth landmarks", "is_suspicious": False}

        # Get mouth bounding box
        mouth_points = np.array([(p["x"], p["y"]) for p in mouth])
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)

        h, w = img.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        mouth_region = img[y_min:y_max, x_min:x_max]
        if mouth_region.size == 0 or mouth_region.shape[0] < 10 or mouth_region.shape[1] < 10:
            return {"score": 50, "detail": "Mouth region too small to analyze", "is_suspicious": False}

        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region

        # Texture analysis — teeth should have distinct texture detail
        # AI-generated teeth often lack fine texture (low Laplacian variance)
        laplacian = cv2.Laplacian(gray_mouth, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))

        # Color uniformity check — AI renders teeth as uniform white
        hsv_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
        saturation = hsv_mouth[:, :, 1].astype(np.float64)
        value = hsv_mouth[:, :, 2].astype(np.float64)

        # Teeth detection: high value, low saturation pixels
        potential_teeth = (value > 150) & (saturation < 60)
        teeth_ratio = float(np.sum(potential_teeth)) / max(potential_teeth.size, 1)

        # Check teeth uniformity (too uniform = AI artifact)
        teeth_pixels = gray_mouth[potential_teeth[:gray_mouth.shape[0], :gray_mouth.shape[1]]] if np.any(potential_teeth[:gray_mouth.shape[0], :gray_mouth.shape[1]]) else np.array([])
        teeth_std = float(np.std(teeth_pixels)) if teeth_pixels.size > 0 else 0

        # ── SPATIAL THRESHOLD: Boundary discontinuity check ───────────────
        # Check if the mouth/lip boundary has a SHARP gradient discontinuity
        # vs surrounding skin. If not, the mouth artifacts are likely just
        # lighting/compression, NOT deepfake artifacts.
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Sample skin texture AROUND the mouth (expand bbox by 50%)
        pad_y = max(1, (y_max - y_min) // 2)
        pad_x = max(1, (x_max - x_min) // 2)
        skin_y1, skin_y2 = max(0, y_min - pad_y), min(h, y_max + pad_y)
        skin_x1, skin_x2 = max(0, x_min - pad_x), min(w, x_max + pad_x)

        skin_region = gray_full[skin_y1:skin_y2, skin_x1:skin_x2]
        mouth_in_skin = gray_full[y_min:y_max, x_min:x_max]

        # Compute gradient at mouth boundary
        sobel_skin = cv2.Sobel(skin_region, cv2.CV_64F, 1, 1, ksize=3)
        sobel_mouth = cv2.Sobel(mouth_in_skin, cv2.CV_64F, 1, 1, ksize=3)

        skin_gradient_mean = float(np.mean(np.abs(sobel_skin)))
        mouth_gradient_mean = float(np.mean(np.abs(sobel_mouth)))

        # Boundary discontinuity: ratio of mouth gradient vs skin gradient
        # If similar (ratio ~1.0), the mouth just has normal lighting/compression
        # If very different (ratio > 2.0 or < 0.4), there's a real discontinuity
        gradient_ratio = mouth_gradient_mean / max(skin_gradient_mean, 0.01)
        has_boundary_discontinuity = gradient_ratio > 2.0 or gradient_ratio < 0.4

        score = 0

        # Low texture variance = smooth/artificial rendering
        if texture_variance < 50:
            score += 35
        elif texture_variance < 150:
            score += 15

        # Very uniform teeth color = AI artifact
        if teeth_ratio > 0.1 and teeth_std < 10:
            score += 30
        elif teeth_ratio > 0.05 and teeth_std < 15:
            score += 15

        # Very high teeth ratio = unrealistic white block
        if teeth_ratio > 0.5:
            score += 20

        score = min(100, score)

        # ── SPATIAL THRESHOLD GATE ────────────────────────────────────────
        # If there is NO boundary discontinuity between mouth and surrounding
        # skin, the mouth anomalies are likely just normal lighting/compression.
        # Suppress the score significantly to prevent false positives.
        if not has_boundary_discontinuity and score > 0:
            score = max(0, round(score * 0.4))  # Suppress by 60%

        if score >= 50:
            detail = "Mouth rendering anomalies with sharp boundary discontinuity — teeth appear unnaturally uniform, strong indicator of AI generation"
            suspicious = True
        elif score >= 25:
            detail = "Some mouth region irregularities — teeth texture may be unusual"
            suspicious = True
        else:
            detail = "Mouth region appears natural with expected texture and detail"
            suspicious = False

        return {
            "score": score,
            "texture_variance": round(texture_variance, 2),
            "teeth_uniformity_std": round(teeth_std, 2),
            "teeth_area_ratio": round(teeth_ratio, 4),
            "gradient_ratio": round(gradient_ratio, 4),
            "has_boundary_discontinuity": has_boundary_discontinuity,
            "detail": detail,
            "is_suspicious": suspicious,
        }

    def _create_annotated_image(
        self, img: np.ndarray, landmarks: dict, results: dict
    ) -> np.ndarray:
        """Create an annotated image with analysis results overlaid."""
        annotated = img.copy()

        if not landmarks:
            return annotated

        # Draw facial landmarks with color coding
        all_lm = landmarks.get("all_landmarks", [])

        # Color-coded regions based on suspicion
        region_colors = {
            "symmetry": (255, 200, 0),     # Cyan
            "eye_reflections": (0, 200, 255),  # Orange
            "boundaries": (0, 255, 100),    # Green
            "mouth": (200, 0, 255),         # Purple
        }

        # Draw jaw outline
        jaw = landmarks.get("jaw", [])
        if len(jaw) > 1:
            color = (0, 0, 255) if results.get("boundaries", {}).get("is_suspicious") else (0, 255, 200)
            pts = np.array([(p["x"], p["y"]) for p in jaw], dtype=np.int32)
            cv2.polylines(annotated, [pts], False, color, 2)

        # Draw eye regions
        for eye_name, color_key in [("left_eye", "eye_reflections"), ("right_eye", "eye_reflections")]:
            eye = landmarks.get(eye_name, [])
            if len(eye) > 2:
                color = (0, 0, 255) if results.get("eye_reflections", {}).get("is_suspicious") else (0, 255, 200)
                pts = np.array([(p["x"], p["y"]) for p in eye], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, color, 2)

        # Draw iris centers
        for iris_name in ["left_iris", "right_iris"]:
            iris = landmarks.get(iris_name, [])
            if iris:
                center = iris[0]
                color = (0, 0, 255) if results.get("eye_reflections", {}).get("is_suspicious") else (0, 255, 200)
                cv2.circle(annotated, (center["x"], center["y"]), 5, color, -1)

        # Draw mouth region
        mouth = landmarks.get("mouth", [])
        if len(mouth) > 5:
            color = (0, 0, 255) if results.get("mouth", {}).get("is_suspicious") else (0, 255, 200)
            pts = np.array([(p["x"], p["y"]) for p in mouth], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.polylines(annotated, [hull], True, color, 2)

        # Add symmetry axis
        nose = landmarks.get("nose", [])
        if nose:
            top = landmarks.get("forehead_center", [{"x": nose[0]["x"], "y": 0}])
            if top:
                x = top[0]["x"]
                color = (0, 0, 255) if results.get("symmetry", {}).get("is_suspicious") else (0, 255, 200)
                cv2.line(annotated, (x, 0), (x, img.shape[0]), color, 1, cv2.LINE_AA)

        return annotated

    def analyze(self, img_bytes: bytes) -> Dict[str, Any]:
        """
        Full face forensics analysis pipeline.

        Returns:
            dict with per-check results, annotated face image, and overall score.
        """
        img_cv2 = load_image_from_bytes(img_bytes)

        # Detect and extract face
        face_crop, face_bbox = self.face_extractor.extract_primary_face(img_cv2)

        if face_crop is None:
            return {
                "face_detected": False,
                "overall_score": 0,
                "verdict": "No face detected in the image. Face forensics analysis requires a clear face.",
                "symmetry": {"score": 0, "detail": "N/A"},
                "eye_reflections": {"score": 0, "detail": "N/A"},
                "boundaries": {"score": 0, "detail": "N/A"},
                "mouth": {"score": 0, "detail": "N/A"},
                "annotated_image_b64": cv2_to_base64(img_cv2),
            }

        # Get landmarks on face crop
        landmarks = self.face_extractor.get_face_landmarks(face_crop)

        # Run all checks
        symmetry = self._compute_symmetry_score(landmarks, face_crop)
        eye_reflections = self._check_eye_reflections(landmarks, face_crop)
        boundaries = self._check_boundary_artifacts(landmarks, face_crop)
        mouth = self._check_mouth_region(landmarks, face_crop)

        results = {
            "symmetry": symmetry,
            "eye_reflections": eye_reflections,
            "boundaries": boundaries,
            "mouth": mouth,
        }

        # Weighted overall score — Forensic Hierarchy:
        # Boundaries (chin/hairline blending) and eye reflections are the
        # strongest biological signals. Symmetry and mouth are secondary
        # because they frequently trigger false positives on real images.
        weights = {
            "symmetry": 0.15,
            "eye_reflections": 0.30,
            "boundaries": 0.40,
            "mouth": 0.15,
        }

        overall_score = sum(
            results[key]["score"] * weight for key, weight in weights.items()
        )
        overall_score = min(100, round(overall_score, 1))

        # Create annotated image
        annotated = self._create_annotated_image(face_crop, landmarks, results)
        annotated_b64 = cv2_to_base64(annotated)

        # Also draw on full image
        if face_bbox:
            full_annotated = img_cv2.copy()
            x, y, w, h = face_bbox
            color = (0, 0, 255) if overall_score >= 50 else (0, 255, 200)
            cv2.rectangle(full_annotated, (x, y), (x + w, y + h), color, 3)
            label = f"Score: {overall_score}"
            cv2.putText(full_annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            full_annotated_b64 = cv2_to_base64(full_annotated)
        else:
            full_annotated_b64 = cv2_to_base64(img_cv2)

        # Verdict
        suspicious_count = sum(1 for r in results.values() if r.get("is_suspicious"))

        # Boost score when multiple checks flag suspicious
        if suspicious_count >= 3:
            overall_score = max(overall_score, 65)
            overall_score = min(100, overall_score + 10)
        elif suspicious_count >= 2:
            overall_score = max(overall_score, 45)
            overall_score = min(100, overall_score + 5)

        if overall_score >= 55:
            verdict = f"HIGH RISK — {suspicious_count}/4 forensic checks flagged. Multiple biometric inconsistencies suggest AI generation."
        elif overall_score >= 35:
            verdict = f"MODERATE RISK — {suspicious_count}/4 forensic checks flagged. Some facial features show unusual characteristics."
        elif overall_score >= 18:
            verdict = f"LOW RISK — {suspicious_count}/4 forensic checks flagged. Minor anomalies detected but generally consistent with real faces."
        else:
            verdict = "MINIMAL RISK — All facial features appear consistent with real human biology."

        return {
            "face_detected": True,
            "face_bbox": face_bbox,
            "overall_score": overall_score,
            "suspicious_checks": suspicious_count,
            "symmetry": symmetry,
            "eye_reflections": eye_reflections,
            "boundaries": boundaries,
            "mouth": mouth,
            "annotated_face_b64": annotated_b64,
            "annotated_full_b64": full_annotated_b64,
            "verdict": verdict,
        }

    def close(self):
        """Release resources."""
        self.face_extractor.close()
