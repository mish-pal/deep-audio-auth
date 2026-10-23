from scipy.signal import butter, sosfilt
import os
import torch
import torchaudio
import librosa
import numpy as np
from src.utils.helpers import pad_or_truncate
import soundfile as sf

def apply_bandpass_filter(waveform, sample_rate, lowcut=80.0, highcut=7500.0, order=5):
    """
    Apply Butterworth bandpass filter to remove microphone handling rumble (<80Hz)
    and high-frequency aliasing/noise (>7500Hz).
    
    Args:
        waveform (torch.Tensor): Audio waveform of shape (channels, samples) or (1, samples)
        sample_rate (int): Sampling rate
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        order (int): Filter order
        
    Returns:
        torch.Tensor: Filtered waveform
    """
    nyq = 0.5 * sample_rate
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    
    # 2nd-order sections (SOS) for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    
    waveform_np = waveform.numpy()
    filtered_np = sosfilt(sos, waveform_np, axis=-1)
    return torch.tensor(filtered_np, dtype=torch.float32)

def check_audio_quality(waveform, sample_rate):
    """
    Analyze the recording quality, energy distribution, and estimated SNR.
    
    Returns:
        dict: Quality metrics and user-friendly status message.
    """
    wav_np = waveform.squeeze().numpy()
    if len(wav_np) == 0:
        return {"is_valid": False, "status": "Empty audio buffer", "snr_db": 0.0}
        
    max_peak = float(np.max(np.abs(wav_np)))
    rms = float(np.sqrt(np.mean(wav_np ** 2)))
    
    # Frame-wise energy estimation for SNR
    frame_len = int(sample_rate * 0.03)  # 30ms frames
    hop_len = int(sample_rate * 0.01)    # 10ms hop
    
    if len(wav_np) > frame_len:
        frames = librosa.util.frame(wav_np, frame_length=frame_len, hop_length=hop_len)
        frame_energies = np.sum(frames ** 2, axis=0) / frame_len
        
        # Bottom 10% energy as noise floor, top 20% as signal
        noise_floor = np.percentile(frame_energies, 10) + 1e-10
        signal_level = np.percentile(frame_energies, 80) + 1e-10
        snr_db = float(10 * np.log10(signal_level / noise_floor))
    else:
        snr_db = 15.0

    if max_peak < 0.02 or rms < 0.003:
        return {
            "is_valid": False,
            "status": "⚠️ Audio is too quiet / No clear voice detected. Please speak louder or closer to the mic.",
            "snr_db": snr_db,
            "rms": rms,
            "peak": max_peak
        }
    elif max_peak >= 0.99:
        return {
            "is_valid": True,
            "status": "⚠️ Minor audio clipping detected. Moving slightly back from the mic will improve accuracy.",
            "snr_db": snr_db,
            "rms": rms,
            "peak": max_peak
        }
    elif snr_db < 6.0:
        return {
            "is_valid": True,
            "status": f"⚠️ High background noise detected (SNR: {snr_db:.1f} dB). Try recording in a quieter room.",
            "snr_db": snr_db,
            "rms": rms,
            "peak": max_peak
        }
    else:
        return {
            "is_valid": True,
            "status": f"✅ Good Audio Quality (SNR: {snr_db:.1f} dB)",
            "snr_db": snr_db,
            "rms": rms,
            "peak": max_peak
        }

def load_audio(file_path, target_sr=16000):
    """
    Load an audio file using soundfile and resample it if necessary.
    """
    data, sr = sf.read(file_path)
    waveform = torch.tensor(data, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
    else:
        waveform = waveform.T  # Shape: (channels, samples)
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform, target_sr

def trim_silence(waveform, sample_rate, top_db=30):
    """
    Trim silence from the beginning and end of the waveform.
    """
    y = waveform.squeeze(0).numpy()
    if np.max(np.abs(y)) < 1e-4:
        return waveform
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) == 0:
        return waveform
    return torch.tensor(y_trimmed, dtype=torch.float32).unsqueeze(0)

def normalize_audio(waveform, target_rms=0.08, max_peak=0.95):
    """
    Apply RMS loudness normalization with a noise floor safety gate.
    Prevents background room noise from being scaled up into fake vocoder patterns.
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val < 1e-5:
        return waveform
        
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    if current_rms < 1e-4:
        # Near-silent audio; do not scale aggressively
        return waveform
        
    # Scale by RMS but ensure peak does not exceed max_peak
    scale = target_rms / (current_rms + 1e-8)
    if max_val * scale > max_peak:
        scale = max_peak / (max_val + 1e-8)
        
    return waveform * scale

def load_and_preprocess_audio(file_path, config, apply_filter=True):
    """
    Full preprocessing pipeline for raw audio signals.
    """
    target_sr = config["audio"]["sample_rate"]
    duration = config["audio"]["duration"]
    trim_db = config["audio"]["trim_db"]
    
    # 1. Load and resample
    waveform, sr = load_audio(file_path, target_sr=target_sr)
    
    # 2. Apply bandpass filter (removes mic rumble and ultrasonic noise)
    if apply_filter:
        waveform = apply_bandpass_filter(waveform, target_sr)
    
    # 3. Trim silence
    if config["audio"].get("trim_silence", True):
        waveform = trim_silence(waveform, sr, top_db=trim_db)
        
    # 4. RMS Loudness Normalization
    waveform = normalize_audio(waveform)
    
    # 5. Pad or truncate to target duration
    target_samples = int(duration * target_sr)
    waveform = pad_or_truncate(waveform, target_samples, dim=-1)
    
    return waveform, target_sr

