import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import time
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

from src.utils.config import load_config
from src.features.preprocessing import (
    load_and_preprocess_audio,
    normalize_audio,
    trim_silence,
    apply_bandpass_filter,
    check_audio_quality
)
from src.features.lfcc import extract_lfcc
from src.features.cqcc import extract_cqcc
from src.models.mbf_net import MBFNet
from src.models.losses import ArcFaceLoss
from src.utils.helpers import pad_or_truncate

# Set Streamlit Page Config
st.set_page_config(
    page_title="Real-Time Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d0f13;
        color: #e5e7eb;
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
    }
    
    /* Premium Header styling */
    .header-container {
        padding: 2.5rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-radius: 16px;
        border: 1px solid #312e81;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        font-weight: 300;
    }
    
    /* Card Container style */
    .custom-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    
    /* Results callouts */
    .result-callout {
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .result-authentic {
        background-color: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        color: #10b981;
    }
    
    .result-deepfake {
        background-color: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4848;
        color: #ef4848;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

@st.cache_resource
def get_device(device_name):
    """Detect available device for computation."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)

@st.cache_resource
def load_detection_model(checkpoint_path, device_name):
    """Load model and loss checkpoints from disk."""
    config = load_config()
    device = get_device(device_name)
    
    if not os.path.exists(checkpoint_path):
        return None, None, config, device
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize Model and load state dict
    model = MBFNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Initialize ArcFace weights
    arcface = ArcFaceLoss(
        in_features=config["model"]["classifier"]["hidden_dim"],
        num_classes=2,
        s=30.0,
        m=0.1
    ).to(device)
    arcface.load_state_dict(checkpoint["arcface_state_dict"])
    arcface.eval()
    
    return model, arcface, config, device

@st.cache_resource
def load_pretrained_generalizer(device_name):
    """Load pre-trained generalization deepfake voice detector model."""
    device = get_device(device_name)
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    model = AutoModelForAudioClassification.from_pretrained("garystafford/wav2vec2-deepfake-voice-detector").to(device)
    extractor = AutoFeatureExtractor.from_pretrained("garystafford/wav2vec2-deepfake-voice-detector")
    model.eval()
    return model, extractor, device

def compute_saliency_map(model, lfcc, cqcc, raw_waveform, device):
    """
    Computes time-frequency saliency maps (Explainable AI - XAI).
    
    Mathematical Concept: Gradient Saliency
    Saliency map represents S(x) = | \partial P(x) / \partial x |
    It computes the absolute gradient of the predicted deepfake probability with respect
    to the handcrafted input features. Large gradients indicate spectral zones that had
    the greatest impact on shifting the model's decision towards 'Fake'.
    """
    # Clone inputs and set requires_grad = True to track gradients back to features
    lfcc_grad = lfcc.clone().detach().to(device).requires_grad_(True)
    cqcc_grad = cqcc.clone().detach().to(device).requires_grad_(True)
    raw_waveform_device = raw_waveform.to(device)
    
    # Extract features, fuse, and compute prediction probability
    features = model.extract_features(lfcc_grad, cqcc_grad, raw_waveform_device)
    probs = model.classifier_head(features)
    
    # Compute gradients back to input representations
    probs.backward()
    
    # Take absolute value of gradients and average across channels
    lfcc_saliency = torch.abs(lfcc_grad.grad).squeeze(0).squeeze(0).cpu().numpy()
    cqcc_saliency = torch.abs(cqcc_grad.grad).squeeze(0).squeeze(0).cpu().numpy()
    
    # Normalize to [0, 1] range for visual scaling
    if lfcc_saliency.max() > 0:
        lfcc_saliency /= lfcc_saliency.max()
    if cqcc_saliency.max() > 0:
        cqcc_saliency /= cqcc_saliency.max()
        
    return lfcc_saliency, cqcc_saliency

# ----------------------------------------------------
# Main Application Flow
# ----------------------------------------------------

def main():
    # Render Application Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Real-Time Audio Deepfake Detection</div>
        <div class="header-subtitle">
            🎙️ Multi-Feature Deep Learning Framework utilizing Linear Frequency and Constant-Q Cepstral representations fused with Wav2Vec 2.0.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Define paths
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    
    # Sidebar config
    st.sidebar.markdown("### ⚙️ System Controls")
    device_selection = st.sidebar.selectbox("Compute Backend", ["auto", "cuda", "mps", "cpu"])
    
    model_mode = st.sidebar.selectbox(
        "Detection Model", 
        [
            "Dual-Engine Hybrid (MBFNet + Generalizer)",
            "Generalization Model (Pre-trained)",
            "Custom MBFNet (Local Toy Dataset)"
        ]
    )
    
    decision_threshold = st.sidebar.slider(
        "Deepfake Classification Threshold",
        min_value=0.30,
        max_value=0.85,
        value=0.55,
        step=0.02,
        help="Scores at or above this threshold are classified as Deepfake. Set to 0.55 for balanced live detection."
    )
    
    # Load corresponding model and configurations
    config = load_config()
    device = get_device(device_selection)
    
    model, arcface = None, None
    pretrained_model, pretrained_extractor = None, None
    
    if "MBFNet" in model_mode or "Dual-Engine" in model_mode:
        model, arcface, config, device = load_detection_model(checkpoint_path, device_selection)
        if model is None:
            st.sidebar.error("❌ No trained checkpoint found at outputs/checkpoints/best_model.pth. Please run src/train.py first.")
        else:
            st.sidebar.success("✅ MBFNet Model Loaded!")
            
    if "Generalization" in model_mode or "Dual-Engine" in model_mode:
        with st.spinner("Loading generalizer model..."):
            pretrained_model, pretrained_extractor, device = load_pretrained_generalizer(device_selection)
        st.sidebar.success("✅ Generalizer Loaded!")
        
    # App State Initialization for History Log
    if "history" not in st.session_state:
        st.session_state["history"] = []
        
    # Input panel layout
    col1, col2 = st.columns(2)
    
    audio_path = None
    input_source = None
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📂 Upload Audio File")
            uploaded_file = st.file_uploader("Select a WAV or MP3 audio file", type=["wav", "mp3"])
            if uploaded_file is not None:
                # Save uploaded bytes to temp file
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                audio_path = temp_file_path
                input_source = uploaded_file.name
        
    with col2:
        with st.container(border=True):
            st.markdown("### 🎤 Record via Microphone")
            st.markdown("Click the recorder widget below and speak (up to 5 seconds):")
            
            # Audio recorder component
            recorded_audio_bytes = audio_recorder(
                text="Click to start / stop recording",
                recording_color="#ef4444",
                neutral_color="#6366f1",
                icon_size="2x"
            )
            
            if recorded_audio_bytes is not None:
                # Save recorded bytes to temp file
                temp_file_path = os.path.join(tempfile.gettempdir(), "mic_recording.wav")
                with open(temp_file_path, "wb") as f:
                    f.write(recorded_audio_bytes)
                audio_path = temp_file_path
                input_source = f"Microphone_Record_{datetime.now().strftime('%H%M%S')}.wav"
                st.audio(recorded_audio_bytes, format="audio/wav")
        
    if audio_path is not None:
        # Run prediction button
        if st.button("🚀 Analyze Audio Sample", type="primary", use_container_width=True):
            with st.spinner("Processing audio, extracting spectral coefficients, and querying neural nodes..."):
                start_time = time.time()
                
                # 1. Loading & standard Preprocessing
                target_sr = config["audio"]["sample_rate"]
                target_duration = config["audio"]["duration"]
                trim_db = config["audio"]["trim_db"]
                
                # Load raw file
                import soundfile as sf
                raw_data, orig_sr = sf.read(audio_path)
                waveform = torch.tensor(raw_data, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if orig_sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
                    waveform = resampler(waveform)
                
                # Apply bandpass filter to eliminate mic rumble & ultrasonic artifacts
                waveform = apply_bandpass_filter(waveform, target_sr)
                
                # Analyze signal quality and SNR
                quality_info = check_audio_quality(waveform, target_sr)
                
                # Trim silence
                waveform = trim_silence(waveform, target_sr, top_db=trim_db)
                # RMS Loudness Normalization with noise floor gate
                waveform = normalize_audio(waveform, target_rms=0.08, max_peak=0.95)
                # Pad/truncate raw waveform
                target_samples = int(target_duration * target_sr)
                preprocessed_waveform = pad_or_truncate(waveform, target_samples, dim=-1)
                
                # 2. Extract Features (Unsqueeze(0) adds a batch dimension for the model)
                lfcc_feat = extract_lfcc(preprocessed_waveform, target_sr, config).unsqueeze(0).to(device)
                cqcc_feat = extract_cqcc(preprocessed_waveform, target_sr, config).unsqueeze(0).to(device)
                raw_waveform_tensor = preprocessed_waveform.unsqueeze(0).to(device) # shape: (1, 1, samples)
                
                # 3. Model Forward Pass
                mbfnet_prob = None
                generalizer_prob = None
                lfcc_saliency, cqcc_saliency = None, None
                
                # Compute MBFNet prediction if selected
                if ("MBFNet" in model_mode or "Dual-Engine" in model_mode) and model is not None:
                    with torch.no_grad():
                        bottleneck = model.extract_features(lfcc_feat, cqcc_feat, raw_waveform_tensor.squeeze(1))
                        clf_prob = model.classifier_head(bottleneck).squeeze().item()
                        
                        w_norm = nn.functional.normalize(arcface.weight)
                        f_norm = nn.functional.normalize(bottleneck)
                        cosine_sims = nn.functional.linear(f_norm, w_norm)
                        sim_delta = (cosine_sims[0, 1] - cosine_sims[0, 0]).item()
                        arcface_fake_prob = 1.0 / (1.0 + np.exp(-sim_delta * 3.5))
                        
                        mbfnet_prob = float(np.clip(0.70 * clf_prob + 0.30 * arcface_fake_prob, 0.01, 0.99))
                        
                    # Generate Saliency Maps for Explanations (XAI)
                    lfcc_saliency, cqcc_saliency = compute_saliency_map(
                        model, lfcc_feat, cqcc_feat, raw_waveform_tensor.squeeze(1), device
                    )
                    
                # Compute Generalizer prediction if selected
                if ("Generalization" in model_mode or "Dual-Engine" in model_mode) and pretrained_model is not None:
                    with torch.no_grad():
                        inputs = pretrained_extractor(
                            preprocessed_waveform.squeeze(0).cpu().numpy(),
                            sampling_rate=target_sr,
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        logits = pretrained_model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                        generalizer_prob = probs[0, 1].item()
                
                # Combine predictions based on the selected mode
                if model_mode == "Dual-Engine Hybrid (MBFNet + Generalizer)":
                    if mbfnet_prob is not None and generalizer_prob is not None:
                        deepfake_prob = 0.65 * generalizer_prob + 0.35 * mbfnet_prob
                    elif generalizer_prob is not None:
                        deepfake_prob = generalizer_prob
                    else:
                        deepfake_prob = mbfnet_prob
                elif model_mode == "Generalization Model (Pre-trained)":
                    deepfake_prob = generalizer_prob if generalizer_prob is not None else 0.5
                else:
                    deepfake_prob = mbfnet_prob if mbfnet_prob is not None else 0.5
                    
                authentic_prob = 1.0 - deepfake_prob
                
                latency = time.time() - start_time
                is_fake = deepfake_prob >= decision_threshold
                prediction = "DEEPFAKE (AI Speech)" if is_fake else "AUTHENTIC (Human Speech)"
                
                # Compute confidence scaled relative to the decision threshold
                if is_fake:
                    span = max(1.0 - decision_threshold, 0.01)
                    confidence = float(np.clip(50.0 + ((deepfake_prob - decision_threshold) / span) * 50.0, 50.0, 99.9))
                else:
                    span = max(decision_threshold, 0.01)
                    confidence = float(np.clip(50.0 + ((decision_threshold - deepfake_prob) / span) * 50.0, 50.0, 99.9))
                
                # Save to history log
                st.session_state["history"].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File Name": input_source,
                    "Prediction": prediction.split(" ")[0],
                    "Confidence": f"{confidence:.1f}%",
                    "Latency": f"{latency:.2f}s"
                })
                
                # ----------------------------------------------------
                # Render Results Panel
                # ----------------------------------------------------
                st.markdown("### 📊 Analysis Outcome")
                
                # Audio quality indicator callout
                st.markdown(f'<div style="padding: 0.75rem 1rem; border-radius: 8px; background-color: #1e293b; border: 1px solid #334155; margin-bottom: 1rem; font-size: 0.95rem;">🎙️ <b>Input Signal Quality:</b> {quality_info["status"]}</div>', unsafe_allow_html=True)
                
                # Color code result
                if deepfake_prob >= 0.5:
                    st.markdown(f'<div class="result-callout result-deepfake">⚠️ Prediction: {prediction} (Confidence: {confidence:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-callout result-authentic">✅ Prediction: {prediction} (Confidence: {confidence:.1f}%)</div>', unsafe_allow_html=True)
                    
                # Probability distribution chart
                prob_df = pd.DataFrame({
                    "Category": ["Authentic", "Deepfake"],
                    "Probability": [authentic_prob * 100.0, deepfake_prob * 100.0]
                })
                st.bar_chart(prob_df, x="Category", y="Probability", color="Category")
                
                # ----------------------------------------------------
                # Visualization Dashboard (Spectral & XAI maps)
                # ----------------------------------------------------
                st.markdown("### 📈 Acoustic Waveform & Feature Explanations")
                
                row_col1, row_col2 = st.columns(2)
                
                # Convert raw waveform to numpy for plotting
                wav_np = preprocessed_waveform.squeeze(0).numpy()
                times = np.linspace(0, target_duration, len(wav_np))
                
                with row_col1:
                    # Waveform plot
                    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#111827')
                    ax.set_facecolor('#111827')
                    ax.plot(times, wav_np, color='#6366f1', alpha=0.9, lw=1)
                    ax.set_title("Waveform (Time vs Amplitude)", color='white', fontsize=10)
                    ax.set_xlabel("Time (seconds)", color='#9ca3af', fontsize=8)
                    ax.set_ylabel("Amplitude", color='#9ca3af', fontsize=8)
                    ax.tick_params(colors='#9ca3af', labelsize=8)
                    ax.grid(True, linestyle='--', color='#1f2937', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # LFCC Feature plot
                    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#111827')
                    ax.set_facecolor('#111827')
                    lfcc_np = lfcc_feat.squeeze(0).squeeze(0).cpu().numpy()
                    img = librosa.display.specshow(lfcc_np, sr=target_sr, hop_length=160, x_axis='time', ax=ax, cmap='viridis')
                    ax.set_title("Linear Frequency Cepstrals (LFCC)", color='white', fontsize=10)
                    ax.set_xlabel("Time (seconds)", color='#9ca3af', fontsize=8)
                    ax.set_ylabel("Cepstral Coefficients", color='#9ca3af', fontsize=8)
                    ax.tick_params(colors='#9ca3af', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                with row_col2:
                    # Constant-Q feature plot
                    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#111827')
                    ax.set_facecolor('#111827')
                    cqcc_np = cqcc_feat.squeeze(0).squeeze(0).cpu().numpy()
                    img2 = librosa.display.specshow(cqcc_np, sr=target_sr, hop_length=160, x_axis='time', ax=ax, cmap='magma')
                    ax.set_title("Constant-Q Cepstrals (CQCC)", color='white', fontsize=10)
                    ax.set_xlabel("Time (seconds)", color='#9ca3af', fontsize=8)
                    ax.set_ylabel("Cepstral Coefficients", color='#9ca3af', fontsize=8)
                    ax.tick_params(colors='#9ca3af', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    if lfcc_saliency is not None:
                        # XAI: LFCC Saliency Map
                        fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#111827')
                        ax.set_facecolor('#111827')
                        ax.imshow(lfcc_saliency, aspect='auto', origin='lower', cmap='hot')
                        ax.set_title("Explainable AI: LFCC Saliency Heatmap", color='white', fontsize=10)
                        ax.set_xlabel("Temporal Frame index", color='#9ca3af', fontsize=8)
                        ax.set_ylabel("Coefficient index (Acoustic artifacts)", color='#9ca3af', fontsize=8)
                        ax.tick_params(colors='#9ca3af', labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("ℹ️ Saliency map visualization (XAI) is only active in Custom MBFNet mode.")
                        
                # Explanation Note
                if lfcc_saliency is not None:
                    st.info("💡 **Explainable AI (XAI) Insight:** The saliency map highlights regions where the model detected anomalies. Brighter (hotter) regions indicate specific acoustic frames and cepstral coefficients that heavily influenced the model's classification. For example, high-frequency grid patches in synthetic speech reflect vocoder interpolation patterns.")

    # ----------------------------------------------------
    # Prediction History Panel
    # ----------------------------------------------------
    if len(st.session_state["history"]) > 0:
        st.markdown("### 🕒 Prediction Session Log")
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()
