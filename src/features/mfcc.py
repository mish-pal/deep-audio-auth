import torchaudio
import torch
from src.utils.helpers import pad_or_truncate

def extract_mfcc(waveform, sample_rate, config):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCC) from raw audio waveform.
    
    Acoustic Concept:
    MFCCs model the frequency sensitivity of the human ear using the Mel scale:
    Mel(f) = 2595 * log10(1 + f/700)
    This provides high frequency resolution at lower frequencies (where human vocal folds
    produce formants) and coarser resolution at higher frequencies. It is excellent for
    capturing speaker identity, vocal tract resonance, and acoustic signatures.
    
    Mathematical Steps:
    1. STFT of raw waveform: X(t, f)
    2. Map spectral bins to Mel-scale filter banks.
    3. Logarithmic scaling: Log_Mel = log10(Mel_Spectrum)
    4. Discrete Cosine Transform: MFCC = DCT(Log_Mel)
    """
    max_steps = config["audio"]["max_time_steps"]
    
    # Configure MFCC transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={
            "n_mels": 40,
            "n_fft": 512,
            "hop_length": 160
        }
    )
    
    # Compute MFCC
    mfcc = mfcc_transform(waveform)
    
    # Pad or truncate
    mfcc_padded = pad_or_truncate(mfcc, max_steps, dim=-1)
    
    return mfcc_padded
