import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, confusion_matrix

def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER) and the corresponding decision threshold.
    
    Mathematical Concept:
    EER is the location where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
    FAR(t) = P(Score >= t | Genuine = False)
    FRR(t) = P(Score < t | Genuine = True)
    
    Using scikit-learn's ROC curve:
    - fpr (False Positive Rate) represents FAR.
    - tpr (True Positive Rate) represents 1 - FRR.
    - So, FRR = 1 - tpr.
    
    We search for the index k where FAR[k] ≈ FRR[k], i.e., fpr[k] ≈ 1 - tpr[k].
    The intersection is found by minimizing the absolute difference: |fpr[k] - (1 - tpr[k])|.
    """
    # Force numpy arrays
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Compute False Positive Rate (FPR) and True Positive Rate (TPR)
    # pos_label=1 implies class 1 (Deepfake) is the positive class.
    # Scores represent the probability or cosine margin logit of being a Deepfake.
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # FRR = 1 - TPR
    fnr = 1.0 - tpr
    
    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.absolute(fpr - fnr))
    
    # Equal Error Rate is the average of FPR and FNR at this point
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    
    return eer, threshold

def compute_all_metrics(labels, scores, threshold=0.5):
    """
    Calculate classification metrics: EER, AUC, Accuracy, Precision, Recall, F1, and Confusion Matrix.
    
    Args:
        labels (list/np.ndarray): Binary target labels (0=Authentic, 1=Deepfake).
        scores (list/np.ndarray): Probability scores or raw logits for being a Deepfake.
        threshold (float): Decision threshold for binary classification.
        
    Returns:
        dict: Metric names and computed values.
    """
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Hard predictions based on threshold
    preds = (scores >= threshold).astype(int)
    
    # 1. EER
    eer, eer_threshold = compute_eer(labels, scores)
    
    # 2. ROC AUC
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 3. Accuracy
    acc = accuracy_score(labels, preds)
    
    # 4. Precision, Recall, F1-score (for Deepfake class)
    # pos_label=1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, beta=1.0, average='binary', pos_label=1, zero_division=0
    )
    
    # 5. Confusion Matrix
    # TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": roc_auc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
