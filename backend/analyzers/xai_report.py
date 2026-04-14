"""
DeepShield AI — Explainable AI (XAI) Report Generator

Synthesizes results from all detection layers (ELA, DCT, Face Forensics,
Classifier) into a structured, human-readable report following the XAI
analysis format with:
  - Classification + confidence
  - Key findings
  - Suspicious regions with explanations
  - Technical analysis (texture, lighting, geometry, noise, deepfake artifacts)
  - Final human-friendly explanation
"""

from typing import Dict, Any, List, Optional


class XAIReportGenerator:
    """
    Generates a comprehensive XAI report from multi-layer analysis results.
    Maps raw analyzer outputs to the structured XAI schema.
    """

    def generate(
        self,
        overall: Dict[str, Any],
        ela: Dict[str, Any],
        dct: Dict[str, Any],
        forensics: Dict[str, Any],
        classifier: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the full XAI report from all analyzer results.

        Args:
            overall: Overall verdict, score, risk_level
            ela: ELA analyzer result dict
            dct: DCT analyzer result dict
            forensics: Face forensics analyzer result dict
            classifier: Classifier analyzer result dict

        Returns:
            Structured XAI report dict
        """
        prediction = overall["verdict"]
        confidence = round(classifier.get("confidence", 0), 1)

        key_findings = self._build_key_findings(overall, ela, dct, forensics, classifier)
        suspicious_regions = self._build_suspicious_regions(ela, dct, forensics)
        technical_analysis = self._build_technical_analysis(ela, dct, forensics)
        final_explanation = self._build_final_explanation(
            prediction, confidence, key_findings, forensics
        )

        return {
            "prediction": prediction,
            "confidence": f"{confidence}%",
            "key_findings": key_findings,
            "suspicious_regions": suspicious_regions,
            "technical_analysis": technical_analysis,
            "final_explanation": final_explanation,
        }

    # ── Key Findings ──────────────────────────────────────────────────────

    def _build_key_findings(
        self,
        overall: Dict[str, Any],
        ela: Dict[str, Any],
        dct: Dict[str, Any],
        forensics: Dict[str, Any],
        classifier: Dict[str, Any],
    ) -> List[str]:
        """Extract the most important findings across all analyzers."""
        findings: List[str] = []

        # ── ELA findings ──
        ela_score = ela.get("score", 0)
        face_bg = ela.get("face_vs_background", {})
        noise = ela.get("noise_consistency", {})
        multi_q = ela.get("multi_quality", {})

        if ela_score >= 50:
            findings.append(
                f"Significant compression inconsistencies detected (ELA score: {ela_score}/100)"
            )
        elif ela_score >= 30:
            findings.append(
                f"Moderate compression anomalies found (ELA score: {ela_score}/100)"
            )

        rel_diff = face_bg.get("relative_differential", 0)
        if rel_diff > 0.8:
            findings.append(
                f"Strong ELA mismatch between face and background regions — {face_bg.get('detail', '')}"
            )
        elif rel_diff > 0.3:
            findings.append(
                "Moderate ELA difference between face and background detected"
            )

        noise_cv = noise.get("noise_cv", 0)
        if noise_cv > 0.7:
            findings.append(
                "Highly inconsistent noise pattern across image — strong indicator of synthetic content"
            )
        elif noise_cv > 0.4:
            findings.append(
                "Moderately inconsistent noise pattern — possible AI generation"
            )

        if multi_q and not multi_q.get("is_monotonic", True):
            findings.append(
                "Non-monotonic error pattern across JPEG quality levels — unusual for real photos"
            )

        # ── DCT findings ──
        dct_score = dct.get("score", 0)
        periodic = dct.get("periodic_artifacts", {})
        freq_dist = dct.get("frequency_distribution", {})

        if dct_score >= 50:
            findings.append(
                f"Strong frequency-domain anomalies detected (DCT score: {dct_score}/100)"
            )
        elif dct_score >= 30:
            findings.append(
                f"Some frequency-domain irregularities found (DCT score: {dct_score}/100)"
            )

        if periodic.get("has_periodic_artifacts"):
            findings.append(
                "Periodic grid artifacts detected in frequency spectrum — consistent with GAN upsampling"
            )

        hf_ratio = freq_dist.get("high_freq_ratio", 0)
        if hf_ratio > 0.1:
            findings.append(
                f"Elevated high-frequency energy ratio ({hf_ratio:.4f}) — potential GAN/diffusion fingerprint"
            )

        # ── Face forensics findings ──
        forensics_score = forensics.get("score", 0)
        face_detected = forensics.get("face_detected", False)

        if face_detected:
            sym = forensics.get("checks", {}).get("symmetry", {})
            eye = forensics.get("checks", {}).get("eye_reflections", {})
            bound = forensics.get("checks", {}).get("boundaries", {})
            mouth = forensics.get("checks", {}).get("mouth", {})

            if bound.get("is_suspicious"):
                findings.append(
                    f"Face boundary artifacts detected — {bound.get('detail', 'blending inconsistencies at jawline/hairline')}"
                )
            if eye.get("is_suspicious"):
                findings.append(
                    f"Eye reflection inconsistency — {eye.get('detail', 'light sources differ between eyes')}"
                )
            if sym.get("is_suspicious"):
                findings.append(
                    f"Unusual facial symmetry — {sym.get('detail', 'face is unnaturally symmetric')}"
                )
            if mouth.get("is_suspicious"):
                findings.append(
                    f"Mouth rendering anomaly — {mouth.get('detail', 'teeth/lip texture issues')}"
                )

            suspicious_count = forensics.get("suspicious_checks", 0)
            if suspicious_count >= 3:
                findings.append(
                    f"{suspicious_count}/4 biometric forensic checks flagged — strong multi-signal evidence"
                )
        else:
            findings.append(
                "No face detected — face forensics analysis was limited"
            )

        # ── Cross-layer agreement ──
        reasoning = classifier.get("reasoning", [])
        for r in reasoning:
            if "Multiple detection layers agree" in r:
                findings.append(r)
            if "False positive suppression" in r:
                findings.append(r)

        # ── If nothing significant found ──
        if not findings:
            findings.append(
                "No significant anomalies detected across all analysis layers"
            )

        return findings

    # ── Suspicious Regions ────────────────────────────────────────────────

    def _build_suspicious_regions(
        self,
        ela: Dict[str, Any],
        dct: Dict[str, Any],
        forensics: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Identify specific suspicious regions with explanations."""
        regions: List[Dict[str, str]] = []

        face_detected = forensics.get("face_detected", False)

        if face_detected:
            checks = forensics.get("checks", {})

            # Eye region
            eye = checks.get("eye_reflections", {})
            if eye.get("is_suspicious"):
                hist_corr = eye.get("hist_correlation", "N/A")
                bright_diff = eye.get("brightness_diff", "N/A")
                regions.append({
                    "region": "Eyes (iris/reflection area)",
                    "issue": (
                        f"Light reflections are inconsistent between left and right eyes. "
                        f"Histogram correlation: {hist_corr}, brightness difference: {bright_diff}. "
                        f"Real eyes reflect the same light sources symmetrically."
                    ),
                })

            # Jawline / boundaries
            bound = checks.get("boundaries", {})
            if bound.get("is_suspicious"):
                ratio = bound.get("boundary_to_global_ratio", "N/A")
                regions.append({
                    "region": "Jawline / chin / hairline (face boundaries)",
                    "issue": (
                        f"Blending artifacts detected at face boundaries. "
                        f"Boundary-to-global gradient ratio: {ratio}. "
                        f"AI-generated faces often have blurry or inconsistent edges where the face meets the background."
                    ),
                })

            # Symmetry
            sym = checks.get("symmetry", {})
            if sym.get("is_suspicious"):
                dev = sym.get("avg_deviation", "N/A")
                regions.append({
                    "region": "Full face (symmetry axis)",
                    "issue": (
                        f"Face is unnaturally symmetric (avg deviation: {dev}). "
                        f"Real human faces have natural asymmetry in eye height, eyebrow shape, and nostril width. "
                        f"AI models tend to produce near-perfect bilateral symmetry."
                    ),
                })

            # Mouth / teeth
            mouth = checks.get("mouth", {})
            if mouth.get("is_suspicious"):
                tex_var = mouth.get("texture_variance", "N/A")
                teeth_std = mouth.get("teeth_uniformity_std", "N/A")
                regions.append({
                    "region": "Mouth / teeth area",
                    "issue": (
                        f"Teeth appear unnaturally uniform or mouth texture is too smooth. "
                        f"Texture variance: {tex_var}, teeth color std: {teeth_std}. "
                        f"AI models struggle with individual tooth rendering and gum detail."
                    ),
                })

        # ELA face vs background mismatch
        face_bg = ela.get("face_vs_background", {})
        rel_diff = face_bg.get("relative_differential", 0)
        if rel_diff > 0.3 and face_bg.get("face_detected"):
            regions.append({
                "region": "Face region vs background (ELA differential)",
                "issue": (
                    f"Compression error levels differ significantly between the face and background. "
                    f"Face mean ELA: {face_bg.get('face_mean', 'N/A')}, "
                    f"Background mean ELA: {face_bg.get('bg_mean', 'N/A')}, "
                    f"Relative differential: {rel_diff:.3f}. "
                    f"This suggests the face was generated or pasted separately from the background."
                ),
            })

        # Noise inconsistency
        noise = ela.get("noise_consistency", {})
        noise_cv = noise.get("noise_cv", 0)
        if noise_cv > 0.4:
            regions.append({
                "region": "Full image (noise pattern grid)",
                "issue": (
                    f"Noise levels vary significantly across different regions of the image. "
                    f"Noise coefficient of variation: {noise_cv:.4f}. "
                    f"Real camera sensors produce uniform noise; AI models generate inconsistent noise."
                ),
            })

        # DCT periodic artifacts
        periodic = dct.get("periodic_artifacts", {})
        if periodic.get("has_periodic_artifacts"):
            regions.append({
                "region": "Full image (frequency spectrum)",
                "issue": (
                    "Periodic grid patterns detected in the DCT frequency domain. "
                    "This is a classic fingerprint of GAN upsampling layers that "
                    "generate images at low resolution and upscale them."
                ),
            })

        if not regions:
            regions.append({
                "region": "None identified",
                "issue": "No specific regions showed significant anomalies.",
            })

        return regions

    # ── Technical Analysis ────────────────────────────────────────────────

    def _build_technical_analysis(
        self,
        ela: Dict[str, Any],
        dct: Dict[str, Any],
        forensics: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build detailed technical analysis for each category."""

        return {
            "texture": self._analyze_texture(ela, forensics),
            "lighting": self._analyze_lighting(ela, forensics),
            "geometry": self._analyze_geometry(forensics),
            "noise": self._analyze_noise(ela, dct),
            "deepfake_artifacts": self._analyze_deepfake_artifacts(dct, forensics),
        }

    def _analyze_texture(self, ela: Dict[str, Any], forensics: Dict[str, Any]) -> str:
        """Skin texture consistency analysis."""
        parts = []

        noise = ela.get("noise_consistency", {})
        noise_cv = noise.get("noise_cv", 0)

        if noise_cv > 0.5:
            parts.append(
                f"Skin texture shows inconsistent micro-patterns across different facial regions "
                f"(noise CV: {noise_cv:.4f}). AI-generated skin often lacks realistic pore/hair detail."
            )
        elif noise_cv > 0.3:
            parts.append(
                f"Minor texture inconsistencies detected (noise CV: {noise_cv:.4f}). "
                f"Could indicate subtle smoothing from AI generation or post-processing."
            )
        else:
            parts.append(
                f"Skin texture appears consistent with natural camera-captured imagery "
                f"(noise CV: {noise_cv:.4f}). No unnatural smoothing or blur patches detected."
            )

        # Mouth texture detail
        if forensics.get("face_detected"):
            mouth = forensics.get("checks", {}).get("mouth", {})
            tex_var = mouth.get("texture_variance", 0)
            if isinstance(tex_var, (int, float)) and tex_var < 50:
                parts.append(
                    f"Mouth region has very low texture detail (variance: {tex_var:.1f}), "
                    f"suggesting artificial rendering."
                )

        return " ".join(parts)

    def _analyze_lighting(self, ela: Dict[str, Any], forensics: Dict[str, Any]) -> str:
        """Lighting consistency analysis."""
        parts = []

        # Eye reflections are the strongest lighting consistency signal
        if forensics.get("face_detected"):
            eye = forensics.get("checks", {}).get("eye_reflections", {})
            hist_corr = eye.get("hist_correlation", 1.0)
            bright_diff = eye.get("brightness_diff", 0)

            if eye.get("is_suspicious"):
                parts.append(
                    f"INCONSISTENT — Eye reflections differ between left and right iris "
                    f"(histogram correlation: {hist_corr:.4f}, brightness diff: {bright_diff:.1f}). "
                    f"In real photos, both eyes reflect the same light sources."
                )
            else:
                parts.append(
                    f"Eye reflections appear consistent (histogram correlation: {hist_corr:.4f}). "
                    f"Light source direction matches across both eyes."
                )

        # ELA face-vs-background differential
        face_bg = ela.get("face_vs_background", {})
        rel_diff = face_bg.get("relative_differential", 0)
        if rel_diff > 0.5:
            parts.append(
                f"Face and background have different compression signatures "
                f"(relative differential: {rel_diff:.3f}), suggesting the face was illuminated "
                f"or processed separately from the background."
            )
        else:
            parts.append(
                f"Face and background compression levels are consistent "
                f"(relative differential: {rel_diff:.3f}), no lighting/shadow mismatch detected."
            )

        return " ".join(parts) if parts else "Lighting analysis inconclusive — no face detected."

    def _analyze_geometry(self, forensics: Dict[str, Any]) -> str:
        """Geometric & structural feature analysis."""
        parts = []

        if not forensics.get("face_detected"):
            return "No face detected — geometric analysis could not be performed."

        # Symmetry
        sym = forensics.get("checks", {}).get("symmetry", {})
        dev = sym.get("avg_deviation", None)
        if dev is not None:
            if sym.get("is_suspicious"):
                parts.append(
                    f"Face symmetry is unnaturally high (deviation: {dev:.4f}). "
                    f"Real faces typically have eye-level deviation of 0.05-0.15, "
                    f"but AI-generated faces often measure below 0.03."
                )
            else:
                parts.append(
                    f"Face symmetry is within normal range (deviation: {dev:.4f}), "
                    f"consistent with natural human facial asymmetry."
                )

        # Boundaries (structural edges)
        bound = forensics.get("checks", {}).get("boundaries", {})
        ratio = bound.get("boundary_to_global_ratio", None)
        if ratio is not None:
            if bound.get("is_suspicious"):
                parts.append(
                    f"Face boundary gradients are abnormal (boundary/global ratio: {ratio:.4f}). "
                    f"Jawline and hairline edges show blurring or warping consistent with face-swap blending."
                )
            else:
                parts.append(
                    f"Face boundary edges appear natural (boundary/global ratio: {ratio:.4f}). "
                    f"No warping or distortion detected at jawline, chin, or hairline."
                )

        return " ".join(parts)

    def _analyze_noise(self, ela: Dict[str, Any], dct: Dict[str, Any]) -> str:
        """Frequency & noise analysis."""
        parts = []

        # Noise consistency
        noise = ela.get("noise_consistency", {})
        noise_cv = noise.get("noise_cv", 0)
        parts.append(
            f"Noise uniformity (CV: {noise_cv:.4f}): "
            + noise.get("detail", "Analysis complete.")
        )

        # DCT frequency distribution
        freq = dct.get("frequency_distribution", {})
        hf = freq.get("high_freq_ratio", 0)
        mf = freq.get("mid_freq_ratio", 0)
        hl = freq.get("high_to_low_ratio", 0)

        parts.append(
            f"Frequency spectrum — high-freq ratio: {hf:.4f}, mid-freq ratio: {mf:.4f}, "
            f"high-to-low ratio: {hl:.4f}."
        )

        # Multi-quality ELA
        multi = ela.get("multi_quality", {})
        if multi:
            mono = multi.get("is_monotonic", True)
            var = multi.get("variation_coefficient", 0)
            if not mono:
                parts.append(
                    f"ELA error degradation across JPEG quality levels is non-monotonic "
                    f"(variation: {var:.4f}), inconsistent with single-source photography."
                )
            else:
                parts.append(
                    f"ELA error pattern across quality levels is smooth and monotonic "
                    f"(variation: {var:.4f}), consistent with normal JPEG compression."
                )

        # DCT periodic artifacts
        periodic = dct.get("periodic_artifacts", {})
        if periodic.get("has_periodic_artifacts"):
            parts.append(
                f"GAN fingerprint detected — periodic energy peaks at diagonal positions "
                f"in block DCT (peak count: {periodic.get('peak_count', 0)})."
            )

        return " ".join(parts)

    def _analyze_deepfake_artifacts(
        self, dct: Dict[str, Any], forensics: Dict[str, Any]
    ) -> str:
        """Deepfake-specific indicator analysis."""
        parts = []

        # Blending boundaries
        if forensics.get("face_detected"):
            bound = forensics.get("checks", {}).get("boundaries", {})
            if bound.get("is_suspicious"):
                parts.append(
                    f"BLENDING BOUNDARIES: Soft/inconsistent gradients detected at face edges "
                    f"(gradient std: {bound.get('gradient_std', 'N/A')}). "
                    f"This is the primary indicator of face-swap or face-generation algorithms."
                )
            else:
                parts.append(
                    "BLENDING BOUNDARIES: Face edges show natural, sharp transitions — no blending artifacts."
                )

            # Mouth warping
            mouth = forensics.get("checks", {}).get("mouth", {})
            if mouth.get("is_suspicious"):
                parts.append(
                    f"WARPING/MORPHING: Mouth region shows rendering anomalies "
                    f"(texture variance: {mouth.get('texture_variance', 'N/A')}). "
                    f"AI-generated teeth and lips often exhibit unnatural uniformity."
                )
        else:
            parts.append(
                "FACE NOT DETECTED: Deepfake-specific face analysis could not be performed."
            )

        # GAN/diffusion fingerprints from DCT
        periodic = dct.get("periodic_artifacts", {})
        dct_score = dct.get("score", 0)
        if periodic.get("has_periodic_artifacts"):
            parts.append(
                "GAN FINGERPRINTS: Periodic spectral artifacts confirm generative model upsampling patterns."
            )
        elif dct_score >= 40:
            parts.append(
                f"FREQUENCY ANOMALIES: Unusual spectral distribution (DCT score: {dct_score}). "
                f"Could indicate diffusion model post-processing."
            )
        else:
            parts.append(
                "No GAN/diffusion model spectral fingerprints detected in the frequency domain."
            )

        # Background consistency
        freq = dct.get("frequency_distribution", {})
        hl = freq.get("high_to_low_ratio", 0)
        if hl > 0.3:
            parts.append(
                f"BACKGROUND: Elevated high-to-low frequency ratio ({hl:.4f}) suggests "
                f"the background may have been synthetically generated or heavily processed."
            )

        return " ".join(parts)

    # ── Final Explanation ─────────────────────────────────────────────────

    def _build_final_explanation(
        self,
        prediction: str,
        confidence: float,
        key_findings: List[str],
        forensics: Dict[str, Any],
    ) -> str:
        """Build a simple, human-friendly explanation."""

        if prediction == "FAKE":
            opening = (
                f"This image is classified as FAKE with {confidence}% confidence. "
            )
            if forensics.get("face_detected"):
                suspicious = forensics.get("suspicious_checks", 0)
                opening += (
                    f"Our analysis detected {suspicious} out of 4 facial biometric anomalies. "
                )

            # Pick the top 3 most important findings
            top_findings = key_findings[:3]
            details = "The main red flags are: " + "; ".join(
                f.lower().rstrip(".") for f in top_findings
            ) + ". "

            closing = (
                "In simple terms, the image shows signs that it was created or modified by AI software. "
                "The face region shows digital artifacts that don't match what we'd expect from a real camera photograph."
            )
            return opening + details + closing

        elif prediction == "SUSPICIOUS":
            opening = (
                f"This image is classified as SUSPICIOUS with {confidence}% confidence. "
            )
            details = (
                "Some detection layers flagged minor anomalies, but evidence is not strong enough "
                "to definitively classify this as fake. "
            )
            if key_findings:
                details += "Points of concern include: " + "; ".join(
                    f.lower().rstrip(".") for f in key_findings[:2]
                ) + ". "

            closing = (
                "The image may have been lightly edited or re-compressed through social media, "
                "which can trigger some of our sensors. Further investigation is recommended."
            )
            return opening + details + closing

        elif prediction == "LIKELY REAL":
            opening = (
                f"This image is classified as LIKELY REAL with {confidence}% confidence. "
            )
            details = (
                "Most detection layers show normal results, with only very minor anomalies "
                "that are commonly caused by JPEG compression or social media re-encoding. "
            )
            closing = (
                "The pixel patterns, noise distribution, and facial features all appear consistent "
                "with a real camera-captured photograph."
            )
            return opening + details + closing

        else:  # REAL
            opening = (
                f"This image is classified as REAL with {confidence}% confidence. "
            )
            details = (
                "All forensic analysis layers — compression analysis, frequency domain inspection, "
                "facial biometric checks, and noise pattern analysis — show results consistent "
                "with an authentic, unmanipulated photograph. "
            )
            closing = (
                "The image exhibits natural noise patterns, consistent lighting across the face, "
                "normal facial asymmetry, and no blending artifacts. "
                "There is no significant evidence of AI generation or digital manipulation."
            )
            return opening + details + closing
