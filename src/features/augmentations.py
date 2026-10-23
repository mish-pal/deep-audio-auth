import torch
import numpy as np
import torchaudio
import torchaudio.functional as F

class AudioAugmentor:
    """
    Apply mathematical data augmentations to raw audio waveforms during training.
    
    Acoustic & Mathematical Concepts:
    1. Background Noise: Models environment channel noise. Added using Gaussian white noise
       scaled to achieve a target Signal-to-Noise Ratio (SNR) in decibels (dB).
       Formula: P_noise = P_signal / (10^(SNR/10))
    2. Reverberation: Models acoustic reflections in physical rooms. Simulated using a
       comb-filter-like loop that superimposes decayed, delayed echoes of the input waveform.
    3. MP3 Compression Simulation: Models high-frequency cutoff and bit-depth quantization
       caused by lossy codecs. Simulated using low-pass biquad filtering and uniform rounding.
    4. Pitch Shifting: Frequency domain modification without changing duration. Performs
       STFT, scales phase updates (phase vocoder), and performs inverse STFT.
    5. Time Stretching (Speed change): Adjusts playback rate. Accomplished by resampling
       the waveform to a virtual rate and restoring original duration via padding/truncation.
    """
    
    def __init__(self, config):
        """
        Args:
            config (dict): Project configuration parameters.
        """
        self.config = config.get("augmentations", {})
        self.enabled = self.config.get("enabled", False)
        
        self.noise_cfg = self.config.get("noise", {})
        self.reverb_cfg = self.config.get("reverb", {})
        self.mp3_cfg = self.config.get("mp3_simulation", {})
        self.pitch_cfg = self.config.get("pitch_shift", {})
        self.stretch_cfg = self.config.get("time_stretch", {})

    def add_background_noise(self, waveform):
        """Inject Gaussian noise with a random SNR."""
        if not self.noise_cfg.get("enabled", False):
            return waveform
            
        snr_db = np.random.uniform(
            self.noise_cfg.get("min_snr_db", 15),
            self.noise_cfg.get("max_snr_db", 30)
        )
        
        # Calculate signal power (Mean Square amplitude)
        # P_signal = 1/N * sum_{i=1}^N x_i^2
        sig_power = torch.mean(waveform ** 2)
        if sig_power == 0:
            return waveform
            
        # Calculate target noise power: P_noise = P_signal / 10^(SNR/10)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        
        # Generate Gaussian noise: G_noise ~ N(0, P_noise)
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def apply_reverberation(self, waveform, sample_rate=16000):
        """Simulate multi-bounce echo/room reverberation using decay equations."""
        if not self.reverb_cfg.get("enabled", False):
            return waveform
            
        delay_ms = np.random.uniform(
            self.reverb_cfg.get("min_delay_ms", 5),
            self.reverb_cfg.get("max_delay_ms", 30)
        )
        decay = self.reverb_cfg.get("decay", 0.4)
        
        # Convert delay from milliseconds to discrete samples: D = delay_ms * sr / 1000
        delay_samples = int(delay_ms * sample_rate / 1000.0)
        if delay_samples <= 0:
            return waveform
            
        # Create decayed delayed echo signals (3 bounces)
        # y(t) = x(t) + alpha * x(t - D) + alpha^2 * x(t - 2D) + alpha^3 * x(t - 3D)
        out = waveform.clone()
        for bounce in range(1, 4):
            total_delay = bounce * delay_samples
            if total_delay >= waveform.shape[1]:
                break
            decay_factor = decay ** bounce
            out[:, total_delay:] += decay_factor * waveform[:, :-total_delay]
            
        return out

    def simulate_mp3_compression(self, waveform, sample_rate=16000):
        """Simulate lossy MP3 compression artifacts."""
        if not self.mp3_cfg.get("enabled", False):
            return waveform
            
        # 1. Apply a random lowpass filter to simulate high-frequency cutoffs
        # The cutoff frequency must be strictly less than the Nyquist frequency (sample_rate / 2)
        # to prevent digital filter instability and NaN values.
        max_cutoff = min(12000, sample_rate // 2 - 100)
        cutoff_freq = np.random.uniform(4000, max_cutoff)
        try:
            waveform = F.lowpass_biquad(waveform, sample_rate, cutoff_freq)
        except Exception:
            pass
            
        # 2. Apply amplitude quantization to simulate lossy quantization noise
        kbps = np.random.uniform(
            self.mp3_cfg.get("min_kbps", 32),
            self.mp3_cfg.get("max_kbps", 128)
        )
        # Higher bitrates map to finer quantization levels
        levels = int(64 + (kbps - 32) * 2)
        waveform = torch.round(waveform * levels) / levels
        
        return waveform

    def apply_pitch_shift(self, waveform, sample_rate=16000):
        """Apply random pitch shift perturbation using phase vocoder algorithm."""
        if not self.pitch_cfg.get("enabled", False):
            return waveform
            
        max_steps = self.pitch_cfg.get("max_steps", 2.0)
        steps = np.random.uniform(-max_steps, max_steps)
        
        try:
            return F.pitch_shift(waveform, sample_rate, steps)
        except Exception:
            return waveform

    def apply_time_stretch(self, waveform, sample_rate=16000):
        """Apply speed perturbation (resampling-based stretch)."""
        if not self.stretch_cfg.get("enabled", False):
            return waveform
            
        min_rate = self.stretch_cfg.get("min_rate", 0.9)
        max_rate = self.stretch_cfg.get("max_rate", 1.1)
        rate = np.random.uniform(min_rate, max_rate)
        
        if rate == 1.0:
            return waveform
            
        orig_length = waveform.shape[1]
        
        try:
            # Resample audio to scale duration
            # e.g., resampling from 16kHz to 17.6kHz speeds up playback by 1.1x
            stretched = F.resample(waveform, orig_freq=sample_rate, new_freq=int(sample_rate * rate))
            
            # Align back to original length (pad or truncate)
            if stretched.shape[1] > orig_length:
                return stretched[:, :orig_length]
            elif stretched.shape[1] < orig_length:
                pad_amount = orig_length - stretched.shape[1]
                return torch.nn.functional.pad(stretched, (0, pad_amount))
            return stretched
        except Exception:
            return waveform

    def augment(self, waveform, sample_rate=16000):
        """
        Apply a random subset of enabled augmentations to the waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform of shape (1, samples)
            sample_rate (int): Sampling rate
            
        Returns:
            torch.Tensor: Augmented audio waveform.
        """
        if not self.enabled:
            return waveform
            
        # Ensure input values lie in [-1, 1] range to avoid floating overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Randomly apply each augmentation independently
        if np.random.random() < 0.5:
            waveform = self.add_background_noise(waveform)
        if np.random.random() < 0.5:
            waveform = self.apply_reverberation(waveform, sample_rate)
        if np.random.random() < 0.5:
            waveform = self.simulate_mp3_compression(waveform, sample_rate)
        if np.random.random() < 0.3:
            waveform = self.apply_pitch_shift(waveform, sample_rate)
        if np.random.random() < 0.3:
            waveform = self.apply_time_stretch(waveform, sample_rate)
            
        return torch.clamp(waveform, -1.0, 1.0)
