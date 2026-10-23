import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    
from tqdm import tqdm
import numpy as np

from src.utils.config import load_config
from src.data.dataset import DeepfakeAudioDataset
from src.models.mbf_net import MBFNet
from src.models.losses import ArcFaceLoss
from src.utils.metrics import compute_eer

def set_seed(seed):
    """
    Ensure reproducibility by setting seeds for all pseudo-random generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, arcface, dataloader, optimizer, scaler, criterion, device):
    """
    Runs a single training epoch with mixed-precision arithmetic.
    """
    model.train()
    arcface.train()
    
    total_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        lfcc = batch["lfcc"].to(device)
        cqcc = batch["cqcc"].to(device)
        raw_waveform = batch["raw_waveform"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Context
        with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
            # 1. Extract bottleneck features from MBFNet
            features = model.extract_features(lfcc, cqcc, raw_waveform)
            
            # 2. Compute ArcFace logits
            logits = arcface(features, labels)
            
            # 3. Calculate ArcFace cross-entropy loss
            loss_arcface = criterion(logits, labels)
            
            # 4. Calculate Binary Cross Entropy loss on Classifier Head
            # model.classifier_head outputs shape (batch, 1) representing probability of class 1 (Deepfake)
            clf_probs = model.classifier_head(features).squeeze(-1)
            loss_clf = nn.functional.binary_cross_entropy(clf_probs, labels.float())
            
            # Combined Loss
            loss = loss_arcface + loss_clf
            
        # Backpropagation scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def validate(model, arcface, dataloader, criterion, device):
    """
    Runs evaluation on the validation split and returns metrics.
    """
    model.eval()
    arcface.eval()
    
    total_loss = 0.0
    all_labels = []
    all_scores_arcface = []
    all_scores_clf = []
    
    with torch.no_grad():
        for batch in dataloader:
            lfcc = batch["lfcc"].to(device)
            cqcc = batch["cqcc"].to(device)
            raw_waveform = batch["raw_waveform"].to(device)
            labels = batch["label"].to(device)
            
            # 1. Extract features
            features = model.extract_features(lfcc, cqcc, raw_waveform)
            
            # 2. Get standard ArcFace logits for loss computation
            logits = arcface(features, labels)
            loss_arcface = criterion(logits, labels)
            
            clf_probs = model.classifier_head(features).squeeze(-1)
            loss_clf = nn.functional.binary_cross_entropy(clf_probs, labels.float())
            
            loss = loss_arcface + loss_clf
            total_loss += loss.item()
            
            # 3. Compute Deepfake probability using Cosine Similarity (ArcFace)
            w_norm = nn.functional.normalize(arcface.weight)
            f_norm = nn.functional.normalize(features)
            cosine_sims = nn.functional.linear(f_norm, w_norm)  # Shape: (batch, 2)
            
            probs = torch.softmax(cosine_sims * arcface.s, dim=-1)
            fake_probs = probs[:, 1].cpu().numpy()  # Class 1 is Deepfake
            
            all_labels.extend(labels.cpu().numpy())
            all_scores_arcface.extend(fake_probs)
            all_scores_clf.extend(clf_probs.cpu().numpy())
            
    val_loss = total_loss / len(dataloader)
    
    # Compute Equal Error Rate (EER) and Accuracy for ArcFace Similarity method
    val_eer_arcface, _ = compute_eer(all_labels, all_scores_arcface)
    preds_arcface = (np.array(all_scores_arcface) >= 0.5).astype(int)
    val_acc_arcface = np.mean(preds_arcface == np.array(all_labels))
    
    # Compute Equal Error Rate (EER) and Accuracy for Classifier Head method
    val_eer_clf, _ = compute_eer(all_labels, all_scores_clf)
    preds_clf = (np.array(all_scores_clf) >= 0.5).astype(int)
    val_acc_clf = np.mean(preds_clf == np.array(all_labels))
    
    return val_loss, val_eer_arcface, val_acc_arcface, val_eer_clf, val_acc_clf

def main():
    # Load parameters
    config = load_config()
    set_seed(config["training"]["seed"])
    
    # Device setup
    device_cfg = config["model"]["device"]
    if device_cfg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_cfg)
    print(f"Training using device: {device}")
    
    # Initialize Dataset and Loaders
    processed_dir = config["paths"]["processed_dir"]
    train_dataset = DeepfakeAudioDataset(
        csv_path=os.path.join(processed_dir, "train.csv"),
        config=config,
        augment=True
    )
    val_dataset = DeepfakeAudioDataset(
        csv_path=os.path.join(processed_dir, "val.csv"),
        config=config,
        augment=False
    )
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model and Loss
    model = MBFNet(config).to(device)
    
    # ArcFace Loss Parameters
    # input features: 256 (bottleneck dim), classes: 2, s: 30, m: 0.5
    arcface = ArcFaceLoss(
        in_features=config["model"]["classifier"]["hidden_dim"],
        num_classes=2,
        s=30.0,
        m=0.1
    ).to(device)
    
    # Standard cross entropy computes loss on the margin-modified logits
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (optimizes both model parameters and ArcFace classification centroids)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=device.type != 'cpu')
    
    # Logging
    log_dir = config["paths"]["log_dir"]
    writer = SummaryWriter(log_dir=log_dir) if TENSORBOARD_AVAILABLE else None
    
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_eer = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    
    epochs = config["training"]["epochs"]
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Run Train and Validation loops
        train_loss = train_epoch(model, arcface, train_loader, optimizer, scaler, criterion, device)
        val_loss, val_eer_arcface, val_acc_arcface, val_eer_clf, val_acc_clf = validate(model, arcface, val_loader, criterion, device)
        scheduler.step()
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("EER/Val_ArcFace", val_eer_arcface, epoch)
            writer.add_scalar("Accuracy/Val_ArcFace", val_acc_arcface, epoch)
            writer.add_scalar("EER/Val_Clf", val_eer_clf, epoch)
            writer.add_scalar("Accuracy/Val_Clf", val_acc_clf, epoch)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  -> ArcFace: Val EER: {val_eer_arcface:.4f} ({(val_eer_arcface * 100):.2f}%) | Val Acc: {val_acc_arcface:.4f}")
        print(f"  -> Classifier Head: Val EER: {val_eer_clf:.4f} ({(val_eer_clf * 100):.2f}%) | Val Acc: {val_acc_clf:.4f} | LR: {current_lr:.6f}")
        
        # Checkpoint saving & Early stopping (based on best Classifier Head EER)
        if val_eer_clf < best_eer:
            best_eer = val_eer_clf
            patience_counter = 0
            
            # Save weights
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "config": config,
                "val_eer": val_eer_clf,
                "val_eer_arcface": val_eer_arcface
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Classifier EER: {val_eer_clf:.4f}, ArcFace EER: {val_eer_arcface:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs of no improvement.")
                break
                
    if writer is not None:
        writer.close()
    print("Training process complete.")

if __name__ == "__main__":
    main()
