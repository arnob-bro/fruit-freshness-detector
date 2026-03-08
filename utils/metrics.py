"""
Metrics Module for Model Evaluation
Provides comprehensive evaluation metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, 
    roc_curve, classification_report
)
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            probas = torch.sigmoid(outputs) if outputs.dim() == 1 else outputs
            preds = (probas > threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds, all_probas)
    return metrics, all_labels, all_preds, all_probas


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Rotten', 'Fresh']
    
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*50 + "\n")


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()



