# Social Guard AI: Deepfake Detection System

> Multi-layered deepfake detection platform with Explainable AI (XAI), combining EfficientNet-B0 deep learning with forensic heuristics.

## Team Members
| Name | Roll No |
|------|---------|
| Abdul Taufique | BTAI-01 |
| Pratik Nannajkar | BTAI-30 |
| Hrishshikesh Nikam | BTAI-31 |
| Altamash Tirandaz | BTAI-58 |

**Department of Artificial Intelligence | Dr. D. Y. Patil Vidyapeeth, Pune**

---

## Overview

Social Guard AI is a deepfake detection platform that uses **4 independent analysis layers** to determine if an image is real or AI-generated/manipulated:

1. **ELA (Error Level Analysis)** — Detects compression inconsistencies in JPEG images
2. **DCT (Frequency Analysis)** — Identifies GAN fingerprints in the frequency domain
3. **Face Forensics** — Checks biometric properties (eye reflections, symmetry, boundaries)
4. **EfficientNet-B0 (Deep Learning)** — Custom-trained CNN on 190K+ images

Results are combined using **80/20 weighted fusion** (80% model + 20% heuristics) and explained using **Grad-CAM heatmaps**.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10 + FastAPI + Uvicorn |
| Deep Learning | PyTorch + torchvision |
| Model | EfficientNet-B0 (4M params, 15.6MB) |
| Image Processing | OpenCV + Pillow |
| Face Detection | MediaPipe (468 landmarks) |
| Frontend | HTML5 + CSS3 + JavaScript |
| Explainability | Grad-CAM (real gradients) |

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── analyzers/
│   │   ├── ela.py                 # Error Level Analysis
│   │   ├── dct.py                 # DCT Frequency Analysis
│   │   ├── face_forensics.py      # Biometric Forensics
│   │   ├── classifier.py          # Score Fusion (80/20)
│   │   ├── model_loader.py        # EfficientNet + Grad-CAM
│   │   └── xai_report.py          # XAI Report Generator
│   ├── models/
│   │   └── efficientnet_deepfake.pth  # Trained model (not in repo)
│   └── utils/
│       ├── image_utils.py
│       └── face_extractor.py
├── frontend/
│   ├── index.html                 # Web interface
│   ├── css/style.css
│   └── js/
├── train_model.py                 # Training script
├── generate_ppt.py                # PPT generator
├── DeepShield_Training_Report.ipynb  # Training report
└── DeepShield_AI_Presentation.pptx   # Project presentation
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model (first time only)
```bash
# Quick training (5K images, ~30 min on CPU)
python train_model.py --max-samples 5000

# Full training (140K images, needs GPU)
python train_model.py
```

### 3. Start the Server
```bash
uvicorn backend.main:app --reload
```

### 4. Open Browser
```
http://localhost:8000
```

Upload any image and the system will analyze it across all 4 detection layers.

## Model Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 83.04% |
| Test Accuracy | 76.64% |
| Fake F1-Score | 0.7700 |
| Real F1-Score | 0.7628 |
| Dataset Size | 190,335 images |
| Training Strategy | Two-phase transfer learning |

## Key Features

- **Multi-layered detection** — 4 independent methods, no single point of failure
- **80/20 Score Fusion** — Trained model leads, heuristics provide safety net
- **Grad-CAM Explainability** — Visual heatmaps showing WHY an image is flagged
- **AI Explanation Panel** — Human-readable per-layer breakdown of the decision
- **Premium UI** — Dark-themed web interface with animations
- **Fast Inference** — < 5 seconds on CPU

## License

This project is for academic purposes.
