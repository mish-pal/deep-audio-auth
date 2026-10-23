import torchaudio
import torch
from src.utils.helpers import pad_or_truncate

def extract_lfcc(waveform, sample_rate, config):
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC) from raw audio waveform.
    
    Acoustic Concept:
    LFCCs use a linear filter bank rather than a Mel-spaced filter bank. This allocates
    equal resolution across all frequency bands. Since speech synthesis models (vocoders)
    often introduce high-frequency artifacts, LFCC is highly effective for anti-spoofing
    as it preserves fine-grained details in the high-frequency spectrum.
    
    Mathematical Steps:
    1. Short-Time Fourier Transform (STFT) of raw waveform:
       X(t, f) = STFT(x(t))
    2. Power Spectrum: P(t, f) = |X(t, f)|^2
    3. Apply Linear Filter Bank: H_lin(t, m) = sum_f P(t, f) * H_m(f)
    4. Logarithmic scale: S(t, m) = log10(H_lin(t, m))
    5. Discrete Cosine Transform (DCT) to decorrelate features:
       LFCC(t, k) = sum_{m=1}^M S(t, m) * cos(pi * k * (2m - 1) / (2M))
    """
    lfcc_cfg = config["features"]["lfcc"]
    max_steps = config["audio"]["max_time_steps"]
    
    # Configure LFCC transform
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=lfcc_cfg["n_lfcc"],
        speckwargs={
            "n_fft": lfcc_cfg["n_fft"],
            "hop_length": lfcc_cfg["hop_length"]
        }
    )
    
    # Compute LFCC
    lfcc = lfcc_transform(waveform)
    
    # Apply Cepstral Mean and Variance Normalization (CMVN) along time axis
    # Mathematical Concept: Cancels out stationary convolutional channel distortions (speakers & mics)
    lfcc_mean = torch.mean(lfcc, dim=-1, keepdim=True)
    lfcc_std = torch.std(lfcc, dim=-1, keepdim=True) + 1e-6
    lfcc_norm = (lfcc - lfcc_mean) / lfcc_std
    
    # Shape of lfcc: (1, n_lfcc, time_steps)
    # Pad or truncate along the last dimension (time_steps)
    lfcc_padded = pad_or_truncate(lfcc_norm, max_steps, dim=-1)
    
    return lfcc_padded
