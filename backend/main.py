"""
DeepShield AI — FastAPI Backend
Main application with API endpoints for deepfake detection.

Endpoints:
  POST /api/analyze       — Full analysis pipeline
  POST /api/analyze/ela   — ELA only
  POST /api/analyze/dct   — DCT only
  POST /api/analyze/forensics — Face forensics only
  GET  /api/health        — Health check
"""

import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from backend.analyzers.ela import ELAAnalyzer
from backend.analyzers.dct import DCTAnalyzer
from backend.analyzers.face_forensics import FaceForensicsAnalyzer
from backend.analyzers.classifier import ClassifierAnalyzer


# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield AI",
    description="Multi-layered deepfake detection API with forensic analysis and explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Analyzer instances ─────────────────────────────────────────────────────────
ela_analyzer = ELAAnalyzer(quality=90, scale=15.0)
dct_analyzer = DCTAnalyzer(block_size=8)
forensics_analyzer = FaceForensicsAnalyzer()
classifier = ClassifierAnalyzer()

# Max upload size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


# ── Helpers ────────────────────────────────────────────────────────────────────
async def validate_and_read(file: UploadFile) -> bytes:
    """Validate uploaded file and return bytes."""
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, WebP, BMP, TIFF.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f}MB). Maximum: 10MB.",
        )

    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")

    return contents


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "DeepShield AI",
        "version": "1.0.0",
        "analyzers": {
            "ela": "active",
            "dct": "active",
            "face_forensics": "active",
            "classifier": "demo_mode" if not classifier.model_loaded else "active",
        },
    }


@app.post("/api/analyze")
async def full_analysis(file: UploadFile = File(...)):
    """
    Full analysis pipeline — runs all detection methods and aggregates results.

    Returns comprehensive JSON with all heatmaps, scores, explanations,
    and an overall verdict.
    """
    start_time = time.time()
    img_bytes = await validate_and_read(file)

    try:
        # Run all analyzers
        ela_result = ela_analyzer.analyze(img_bytes)
        dct_result = dct_analyzer.analyze(img_bytes)
        forensics_result = forensics_analyzer.analyze(img_bytes)

        # Run classifier with other scores as input
        classifier_result = classifier.analyze(
            img_bytes,
            ela_score=ela_result["overall_score"],
            dct_score=dct_result["overall_score"],
            forensics_score=forensics_result["overall_score"],
            forensics_result=forensics_result,
        )

        # Compute overall verdict
        overall_score = classifier_result["combined_score"]

        if overall_score >= 50:
            overall_verdict = "FAKE"
            risk_level = "HIGH"
        elif overall_score >= 35:
            overall_verdict = "SUSPICIOUS"
            risk_level = "MODERATE"
        elif overall_score >= 20:
            overall_verdict = "LIKELY REAL"
            risk_level = "LOW"
        else:
            overall_verdict = "REAL"
            risk_level = "MINIMAL"

        elapsed = round(time.time() - start_time, 2)

        return JSONResponse(
            content={
                "status": "success",
                "analysis_time_seconds": elapsed,
                "filename": file.filename,
                "overall": {
                    "verdict": overall_verdict,
                    "risk_level": risk_level,
                    "score": overall_score,
                    "prediction": classifier_result["prediction"],
                    "confidence": classifier_result["confidence"],
                    "reasoning": classifier_result["reasoning"],
                    "gradcam_b64": classifier_result["gradcam_b64"],
                },
                "ela": {
                    "score": ela_result["overall_score"],
                    "verdict": ela_result["verdict"],
                    "heatmap_b64": ela_result["ela_heatmap_b64"],
                    "overlay_b64": ela_result["ela_overlay_b64"],
                    "stats": {
                        "max_error": ela_result["max_error"],
                        "mean_error": ela_result["mean_error"],
                        "error_std": ela_result["error_std"],
                        "p95_error": ela_result.get("p95_error", 0),
                        "region_variance": ela_result["region_variance"],
                    },
                    "face_vs_background": ela_result.get("face_vs_background", {}),
                    "noise_consistency": ela_result.get("noise_consistency", {}),
                    "multi_quality": ela_result.get("multi_quality", {}),
                },
                "dct": {
                    "score": dct_result["overall_score"],
                    "verdict": dct_result["verdict"],
                    "spectral_map_b64": dct_result["spectral_map_b64"],
                    "spectral_overlay_b64": dct_result["spectral_overlay_b64"],
                    "frequency_distribution": dct_result["frequency_distribution"],
                    "periodic_artifacts": dct_result["periodic_artifacts"],
                },
                "forensics": {
                    "score": forensics_result["overall_score"],
                    "verdict": forensics_result["verdict"],
                    "face_detected": forensics_result["face_detected"],
                    "suspicious_checks": forensics_result.get("suspicious_checks", 0),
                    "annotated_face_b64": forensics_result.get("annotated_face_b64", ""),
                    "annotated_full_b64": forensics_result.get("annotated_full_b64", ""),
                    "checks": {
                        "symmetry": {
                            "score": forensics_result["symmetry"]["score"],
                            "detail": forensics_result["symmetry"]["detail"],
                            "is_suspicious": forensics_result["symmetry"].get("is_suspicious", False),
                        },
                        "eye_reflections": {
                            "score": forensics_result["eye_reflections"]["score"],
                            "detail": forensics_result["eye_reflections"]["detail"],
                            "is_suspicious": forensics_result["eye_reflections"].get("is_suspicious", False),
                        },
                        "boundaries": {
                            "score": forensics_result["boundaries"]["score"],
                            "detail": forensics_result["boundaries"]["detail"],
                            "is_suspicious": forensics_result["boundaries"].get("is_suspicious", False),
                        },
                        "mouth": {
                            "score": forensics_result["mouth"]["score"],
                            "detail": forensics_result["mouth"]["detail"],
                            "is_suspicious": forensics_result["mouth"].get("is_suspicious", False),
                        },
                    },
                },
                "classifier": {
                    "model_loaded": classifier_result["model_loaded"],
                    "model_note": classifier_result.get("model_note", ""),
                },
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/ela")
async def analyze_ela(file: UploadFile = File(...)):
    """Run Error Level Analysis only."""
    img_bytes = await validate_and_read(file)
    try:
        result = ela_analyzer.analyze(img_bytes)
        return JSONResponse(content={"status": "success", "ela": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ELA analysis failed: {str(e)}")


@app.post("/api/analyze/dct")
async def analyze_dct(file: UploadFile = File(...)):
    """Run DCT Frequency Domain Analysis only."""
    img_bytes = await validate_and_read(file)
    try:
        result = dct_analyzer.analyze(img_bytes)
        return JSONResponse(content={"status": "success", "dct": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DCT analysis failed: {str(e)}")


@app.post("/api/analyze/forensics")
async def analyze_forensics(file: UploadFile = File(...)):
    """Run Face Forensics Analysis only."""
    img_bytes = await validate_and_read(file)
    try:
        result = forensics_analyzer.analyze(img_bytes)
        return JSONResponse(content={"status": "success", "forensics": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensics analysis failed: {str(e)}")


# ── Mount frontend (for local development) ────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
