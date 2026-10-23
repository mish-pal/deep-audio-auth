import os
import glob
import pandas as pd
import hashlib
from src.utils.config import load_config

def extract_speaker_id(filename):
    """
    Extract a speaker ID from the filename.
    
    Algorithm:
    1. If the filename contains a pattern like 'p[0-9]+' (VCTK style), extract it.
    2. If it contains 'spk_[0-9]+', extract it.
    3. If no explicit speaker pattern is found, hash the base filename prefix to assign
       a pseudo-speaker ID, or group them based on prefix to prevent artificial overlap.
       For ASVspoof subset files: 'asvspoof2019_train_real_000.wav' -> we hash the name
       without the file index to create groups, or treat each file as a unique speaker
       if no grouping is known.
    
    Mathematical Concept (Hashing):
    Hashing maps arbitrary length strings to fixed-size values. We use MD5 hash truncated
    to 8 characters to assign consistent speaker IDs when they are not explicit.
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Check for VCTK-style speaker IDs (e.g., p225_001.wav -> p225)
    parts = name_without_ext.split("_")
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            return part
        if part.startswith("spk"):
            return part
            
    # Fallback: Hash the prefix of the filename (everything before the last number)
    # E.g., 'asvspoof2019_train_real_000' -> prefix 'asvspoof2019_train_real'
    # This creates a pseudo-speaker group based on prefix
    prefix = "_".join(parts[:-1]) if len(parts) > 1 else name_without_ext
    hash_obj = hashlib.md5(prefix.encode())
    return f"spk_{hash_obj.hexdigest()[:8]}"

def verify_speaker_leakage(train_df, val_df, test_df):
    """
    Check for speaker leakage (overlap) between dataset splits.
    
    Mathematical Concept: Set Intersection
    Let T, V, and S be the sets of speaker IDs in train, validation, and test splits.
    Speaker leakage exists if:
    (T ∩ V) ∪ (T ∩ S) ∪ (V ∩ S) ≠ ∅ (empty set)
    
    Args:
        train_df, val_df, test_df (pd.DataFrame): Dataframes containing 'speaker_id' column.
        
    Returns:
        bool: True if speaker leakage is detected, False otherwise.
    """
    train_spk = set(train_df['speaker_id'].unique())
    val_spk = set(val_df['speaker_id'].unique())
    test_spk = set(test_df['speaker_id'].unique())
    
    leakage_train_val = train_spk.intersection(val_spk)
    leakage_train_test = train_spk.intersection(test_spk)
    leakage_val_test = val_spk.intersection(test_spk)
    
    has_leakage = False
    if leakage_train_val:
        print(f"WARNING: Speaker leakage between Train and Validation: {leakage_train_val}")
        has_leakage = True
    if leakage_train_test:
        print(f"WARNING: Speaker leakage between Train and Test: {leakage_train_test}")
        has_leakage = True
    if leakage_val_test:
        print(f"WARNING: Speaker leakage between Validation and Test: {leakage_val_test}")
        has_leakage = True
        
    if not has_leakage:
        print("SUCCESS: No speaker leakage detected between train, validation, and test splits.")
        
    return has_leakage

def process_split_directory(split_dir):
    """
    Scans a split directory (containing 'real' and 'fake' subfolders) and extracts metadata.
    
    Args:
        split_dir (str): Path to the split directory (e.g. data/raw/dataset/train).
        
    Returns:
        pd.DataFrame: Dataframe containing columns [file_path, label, speaker_id, source_dataset]
    """
    records = []
    
    for label_str, label_val in [("real", 0), ("fake", 1)]:
        folder_path = os.path.join(split_dir, label_str)
        if not os.path.exists(folder_path):
            continue
            
        search_path = os.path.join(folder_path, "*.wav")
        for wav_path in glob.glob(search_path):
            filename = os.path.basename(wav_path)
            speaker_id = extract_speaker_id(filename)
            
            # Infer dataset source from filename prefix
            source_dataset = "asvspoof2019"
            if "wavefake" in filename.lower():
                source_dataset = "wavefake"
            elif "librispeech" in filename.lower():
                source_dataset = "librispeech"
                
            records.append({
                "file_path": os.path.abspath(wav_path),
                "label": label_val,
                "speaker_id": speaker_id,
                "source_dataset": source_dataset
            })
            
    return pd.DataFrame(records)

def process_cross_dataset_directory(eval_dir):
    """
    Scans the cross-dataset evaluation directory (ASVspoof 2021).
    Since ASVspoof 2021 has different structure (all files in one folder),
    we assign labels based on file index (odd=real, even=fake in dry-run/toy setup)
    or read from an external mapping if available.
    
    For the toy dataset in 'dataset/asvspoof2021', the files are labeled
    synthetically: file index even is fake, odd is real.
    """
    records = []
    if not os.path.exists(eval_dir):
        return pd.DataFrame()
        
    search_path = os.path.join(eval_dir, "*.wav")
    for wav_path in sorted(glob.glob(search_path)):
        filename = os.path.basename(wav_path)
        # Parse the index out of filename: 'asv2021_eval_007.wav' -> 7
        try:
            parts = os.path.splitext(filename)[0].split("_")
            idx = int(parts[-1])
        except Exception:
            idx = 0
            
        # Match label assignment used in dry-run: even=fake (1), odd=real (0)
        label_val = 1 if idx % 2 == 0 else 0
        speaker_id = "unknown"
        
        records.append({
            "file_path": os.path.abspath(wav_path),
            "label": label_val,
            "speaker_id": speaker_id,
            "source_dataset": "asvspoof2021"
        })
        
    return pd.DataFrame(records)

def main():
    # Load configuration
    config = load_config()
    
    raw_dir = config["paths"]["raw_dataset_dir"]
    processed_dir = config["paths"]["processed_dir"]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Reading raw dataset from {raw_dir}...")
    
    # Process standard splits
    train_df = process_split_directory(os.path.join(raw_dir, "train"))
    val_df = process_split_directory(os.path.join(raw_dir, "validation"))
    test_df = process_split_directory(os.path.join(raw_dir, "test"))
    
    # Process cross-dataset split
    cross_df = process_cross_dataset_directory(os.path.join(raw_dir, "asvspoof2021"))
    
    print(f"Processed counts:")
    print(f"  Train: {len(train_df)} samples (Real: {len(train_df[train_df['label'] == 0])}, Fake: {len(train_df[train_df['label'] == 1])})")
    print(f"  Val: {len(val_df)} samples (Real: {len(val_df[val_df['label'] == 0])}, Fake: {len(val_df[val_df['label'] == 1])})")
    print(f"  Test (In-Domain): {len(test_df)} samples (Real: {len(test_df[test_df['label'] == 0])}, Fake: {len(test_df[test_df['label'] == 1])})")
    print(f"  Test (Cross-Dataset): {len(cross_df)} samples")
    
    # Check for speaker leakage
    verify_speaker_leakage(train_df, val_df, test_df)
    
    # Save processed CSVs
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_in_domain.csv"), index=False)
    
    if not cross_df.empty:
        cross_df.to_csv(os.path.join(processed_dir, "test_cross_dataset.csv"), index=False)
        print(f"Saved cross-dataset protocol to {os.path.join(processed_dir, 'test_cross_dataset.csv')}")
        
    print(f"Metadata CSVs saved to {processed_dir}")

if __name__ == "__main__":
    main()
