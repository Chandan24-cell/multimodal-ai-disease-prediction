# training/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import logging

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """
    Calculate comprehensive evaluation metrics for multi-label/multi-class predictions.
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        threshold: Decision threshold for multi-label conversion.
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    # 1. Accuracy (Exact match ratio for multi-label, or standard accuracy for multi-class)
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # 2. F1 Score (Macro and Micro)
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    # 3. Confusion Matrix (Per class)
    # Note: For multi-label, this returns a list of 2x2 matrices or you can use multilabel_confusion_matrix
    from sklearn.metrics import multilabel_confusion_matrix
    cm = multilabel_confusion_matrix(y_true, y_pred_binary)
    
    # 4. ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs, average='macro')
    except ValueError:
        roc_auc = 0.0
        logger.warning("Could not calculate ROC-AUC. Ensure all classes are present in y_true.")

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist() # JSON serializable
    }
    
    logger.info(f"Evaluation Metrics: Acc={accuracy:.4f}, F1-Macro={f1_macro:.4f}, ROC-AUC={roc_auc:.4f}")
    return metrics