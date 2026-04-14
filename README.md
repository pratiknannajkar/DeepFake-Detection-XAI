# 🛡️ DeepShield AI — Deepfake Detection Platform

Multi-layered deepfake detection web application that combines forensic analysis, frequency-domain fingerprinting, and deep learning with visual explainability.

## 🔬 Detection Methods

| Method | Type | What It Detects |
|--------|------|-----------------|
| **ELA** (Error Level Analysis) | Pixel Forensics | Compression inconsistencies from image manipulation |
| **DCT** (Discrete Cosine Transform) | Spectral Forensics | GAN/Diffusion upsampling artifacts in frequency domain |
| **Face Forensics** | Biometric Analysis | Eye reflections, facial symmetry, boundary artifacts, teeth rendering |
| **Grad-CAM** | AI / XAI | Visual explanation of which regions triggered the detection |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend
```bash
# From the project root (DeepFake/)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the Frontend
Navigate to `http://localhost:8000` in your browser.

The frontend is served automatically by FastAPI as static files.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Full analysis pipeline (all methods) |
| `POST` | `/api/analyze/ela` | Error Level Analysis only |
| `POST` | `/api/analyze/dct` | DCT Frequency Analysis only |
| `POST` | `/api/analyze/forensics` | Face Forensics only |
| `GET` | `/api/health` | Health check |

### Example cURL
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@path/to/image.jpg"
```

## 🏗️ Architecture

```
DeepFake/
├── backend/
│   ├── main.py              # FastAPI app & endpoints
│   ├── requirements.txt     # Python dependencies
│   ├── analyzers/
│   │   ├── ela.py           # Error Level Analysis
│   │   ├── dct.py           # DCT Frequency Analysis
│   │   ├── face_forensics.py # Biometric heuristic checks
│   │   └── classifier.py   # CNN + Grad-CAM (demo/real mode)
│   ├── utils/
│   │   ├── face_extractor.py # Face detection (OpenCV + MediaPipe)
│   │   └── image_utils.py   # Image processing utilities
│   └── models/              # Pre-trained model weights
├── frontend/
│   ├── index.html           # Main SPA
│   ├── css/style.css        # Premium dark theme
│   └── js/
│       ├── app.js           # Main controller
│       ├── upload.js        # Upload handling
│       ├── results.js       # Results rendering
│       └── animations.js   # Background particles & animations
└── README.md
```

## 🎨 Features

- **Premium Dark UI** — Glassmorphism, neon accents, animated particles
- **Drag & Drop Upload** — With file validation and preview
- **Real-time Analysis** — Processes images through 4 detection layers
- **Interactive Heatmaps** — ELA, DCT spectral, and Grad-CAM overlays
- **Face Forensics** — Checks symmetry, eye reflections, boundaries, mouth
- **Animated Results** — Score ring animation, counter animations, tabbed dashboard
- **Responsive Design** — Works on desktop and mobile

## 🔧 Adding a Real Model

To use a trained deepfake detection model instead of demo mode:

1. Place your model file in `backend/models/`:
   - TensorFlow/Keras: `deepfake_detector.h5`
   - PyTorch: `deepfake_detector.pth`

2. The classifier will automatically detect and load the model.

Recommended architectures: XceptionNet or EfficientNet fine-tuned on FaceForensics++.

## 📚 Tech Stack

- **Backend**: FastAPI, OpenCV, MediaPipe, SciPy, NumPy, Pillow
- **Frontend**: Vanilla HTML/CSS/JS, Canvas API
- **Detection**: ELA, DCT, Face Mesh Landmarks, Grad-CAM

## 📄 License

Built for educational and research purposes. Use responsibly.
