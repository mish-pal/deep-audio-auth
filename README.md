---
title: Real-Time Deepfake Audio Detection
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Real-Time Deepfake Audio Detection

An advanced, production-grade deep learning framework for detecting synthetic and AI-generated speech in real time. Built with **PyTorch**, **Transformers**, and **Streamlit**, featuring **Explainable AI (XAI)** via spectral saliency maps.

---

## 🌟 Key Features

- **Dual-Engine Detection**:
  - **Custom MBFNet (Multi-Branch Fusion Network)**: Combines handcrafted acoustic features with self-supervised speech representations.
  - **Pre-trained Generalizer**: Wav2Vec 2.0 deepfake classifier providing out-of-domain robustness against zero-day TTS/VC algorithms.
  - **Ensemble Mode**: Weighted consensus between both engines for maximum accuracy and minimized False Alarm Rate.
- **Multi-Feature Acoustic Analysis**:
  - **LFCC (Linear Frequency Cepstral Coefficients)**: Captures spectral tilt and high-frequency vocoder artifacts.
  - **CQCC (Constant-Q Cepstral Coefficients)**: Geometrically spaced frequency bins for rich low-frequency fundamental harmonic analysis.
  - **Raw Waveform SSL**: 12-layer Wav2Vec 2.0 with learnable layer-wise attention pooling.
- **Symmetric Cross-Attention Fusion Module (S-CAFM)**:
  - Cross-attends handcrafted acoustic features and SSL contextual representations in both directions.
- **Metric Learning (ArcFace)**:
  - Angular margin penalty ($s=30, m=0.1$) enforces tight intra-class clustering and separated decision boundaries.
- **Explainable AI (XAI)**:
  - Computes gradient saliency maps ($S(x) = |\nabla_{x} P(\text{Fake})|$) to visually highlight anomalous spectral bins and vocoder artifacts.
- **Multi-Modal Audio Input**:
  - File upload supporting `.wav` and `.mp3`.
  - Direct live microphone recording with web audio capture.

---

## 🏗️ System Architecture

```
                 [ Raw Audio Signal / Mic Stream ]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
 [ Handcrafted Pipeline ]                     [ Deep SSL Pipeline ]
   ├── LFCC Extraction                          └── Wav2Vec 2.0
   └── CQCC Extraction                                (12 Hidden Layers)
           │                                           │
    ResNet Backbones                          Layer-Wise Attention Pooling
           │                                           │
    Handcrafted Embedding (256-D)               SSL Embedding (256-D)
           └─────────────────────┬─────────────────────┘
                                 ▼
              [ Symmetric Cross-Attention Fusion (S-CAFM) ]
                                 │
                     Fused Representation (1024-D)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             [ Linear Head ]          [ ArcFace Head ]
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                     [ Calibrated Probability ]
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          Decision (Real / Fake)     Spectral Saliency (XAI)
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.9+
- System audio libraries:
  - **macOS**: `brew install libsndfile ffmpeg git-lfs`
  - **Ubuntu/Debian**: `sudo apt-get install -y libsndfile1 ffmpeg git-lfs`

### 2. Installation
```bash
git clone <your-repo-url>
cd deep_audio_auth

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch App
```bash
streamlit run app.py
```

---

## 📦 Deployment Instructions (Hugging Face Spaces)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set **SDK** to **Streamlit**.
3. Initialize Git LFS and push:
```bash
git lfs install
git remote add origin https://huggingface.co/spaces/<YOUR-USERNAME>/<YOUR-SPACE-NAME>
git push -u origin main
```
The Space will automatically install packages from `packages.txt` and `requirements.txt` and launch `app.py`.

---

## 📄 License
This project is licensed under the MIT License.
