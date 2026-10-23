import librosa
import numpy as np
import torch
from scipy.fftpack import dct
from src.utils.helpers import pad_or_truncate

def extract_cqcc(waveform, sample_rate, config):
    """
    Extract Constant-Q Cepstral Coefficients (CQCC) from raw audio waveform.
    
    Acoustic Concept:
    Unlike Short-Time Fourier Transform (STFT) which uses linearly spaced frequency bins,
    the Constant-Q Transform (CQT) uses geometrically spaced frequency bins. CQT offers:
    - High frequency resolution at low frequencies (essential for capturing pitch/intonation).
    - High time resolution at high frequencies (essential for capturing rapid transients).
    CQCC is exceptionally robust at capturing the subtle spectral phase anomalies and
    repetitive grid structures produced by vocoders.
    
    Mathematical Steps:
    1. CQT calculation:
       X^Q(k, n) = sum_{j=n - N_k/2}^{n + N_k/2} x[j] * w_k[j - n + N_k/2] * e^{-i * 2*pi * f_k * j / f_s}
       where the window length N_k is inversely proportional to frequency: N_k = Q * f_s / f_k.
    2. Power scale log conversion:
       Log_CQT = 20 * log10(|X^Q(k, n)|)
    3. Discrete Cosine Transform (DCT):
       CQCC = DCT(Log_CQT, type=2, norm='ortho')
       We take the first 40 coefficients to retain the envelope features while discarding
       high-quefrency source excitation details.
    """
    cqcc_cfg = config["features"]["cqcc"]
    max_steps = config["audio"]["max_time_steps"]
    
    # librosa.cqt requires a 1D numpy array
    y = waveform.squeeze(0).numpy()
    
    # Compute Constant-Q Transform (CQT)
    cqt = np.abs(librosa.cqt(
        y, 
        sr=sample_rate, 
        hop_length=cqcc_cfg["hop_length"],
        n_bins=cqcc_cfg["n_bins"],
        bins_per_octave=cqcc_cfg["bins_per_octave"]
    ))
    
    # Convert amplitude to Decibels
    log_cqt = librosa.amplitude_to_db(cqt)
    
    # Discrete Cosine Transform (DCT) along the frequency axis
    # axis=0 corresponds to the frequency bins (since librosa.cqt shape is (freq_bins, frames))
    cqcc = dct(log_cqt, type=2, axis=0, norm='ortho')[:40, :]
    
    # Convert back to tensor of shape (1, 40, time_steps)
    cqcc_tensor = torch.tensor(cqcc, dtype=torch.float32).unsqueeze(0)
    
    # Apply Cepstral Mean and Variance Normalization (CMVN) along time axis
    cqcc_mean = torch.mean(cqcc_tensor, dim=-1, keepdim=True)
    cqcc_std = torch.std(cqcc_tensor, dim=-1, keepdim=True) + 1e-6
    cqcc_norm = (cqcc_tensor - cqcc_mean) / cqcc_std
    
    # Pad or truncate along the time dimension
    cqcc_padded = pad_or_truncate(cqcc_norm, max_steps, dim=-1)
    
    return cqcc_padded
