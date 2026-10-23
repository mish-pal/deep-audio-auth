import torch
from torch.utils.data import Dataset
import pandas as pd
from src.features.preprocessing import load_and_preprocess_audio
from src.features.lfcc import extract_lfcc
from src.features.cqcc import extract_cqcc
from src.features.augmentations import AudioAugmentor

class DeepfakeAudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing audio deepfake samples.
    
    Acoustic & Deep Learning Workflow:
    For each sample index, this dataset:
    1. Loads the audio file using the preprocessing pipeline (mono downmix, resampled to 16kHz,
       silence trimmed, normalized, padded/truncated to 4.0 seconds).
    2. If training and augmentations are enabled, applies randomized augmentations in-memory
       (adds noise, codec distortion, pitch shift, reverberation, speed adjustments).
    3. Extracts handcrafted cepstral coefficients:
       - LFCC: linear filterbank response capturing synthetic vocoder spectral noise.
       - CQCC: Constant-Q transform coefficients capturing temporal and phase discontinuities.
    4. Returns a dictionary containing handcrafted features, raw waveform (for Wav2Vec2/SSL),
       and target classification label (0=Authentic, 1=Deepfake).
    """
    
    def __init__(self, csv_path, config, augment=False):
        """
        Args:
            csv_path (str): Path to the processed metadata CSV protocol.
            config (dict): Configuration dictionary loaded from yaml.
            augment (bool): Whether to apply online data augmentations during load.
        """
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.augment = augment and config["augmentations"]["enabled"]
        self.sample_rate = config["audio"]["sample_rate"]
        
        if self.augment:
            self.augmentor = AudioAugmentor(config)
            print(f"Online augmentations enabled for dataset loaded from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        label = int(row["label"])
        speaker_id = row["speaker_id"]
        
        # 1. Load and standard preprocess raw audio waveform
        waveform, sr = load_and_preprocess_audio(file_path, self.config)
        
        # 2. Apply optional online data augmentation (only on raw waveform)
        if self.augment:
            waveform = self.augmentor.augment(waveform, sr)
            
        # 3. Extract Handcrafted Spectral Features from (potentially augmented) waveform
        lfcc = extract_lfcc(waveform, sr, self.config)
        cqcc = extract_cqcc(waveform, sr, self.config)
        
        # 4. Construct outputs
        # Note: Wav2Vec2 expects raw waveform input of shape (samples,) or (1, samples)
        # We squeeze the channel dimension to yield a 1D tensor: (samples,)
        raw_waveform = waveform.squeeze(0)
        
        return {
            "lfcc": lfcc,                    # Shape: (1, 40, max_time_steps)
            "cqcc": cqcc,                    # Shape: (1, 40, max_time_steps)
            "raw_waveform": raw_waveform,    # Shape: (samples,)
            "label": torch.tensor(label, dtype=torch.long),
            "file_path": file_path,
            "speaker_id": speaker_id
        }
