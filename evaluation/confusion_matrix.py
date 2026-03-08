"""
Confusion Matrix Visualization
Standalone script for generating confusion matrices
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import get_loaders
from utils.metrics import evaluate_model
from models.cnn_scratch import get_model as get_cnn_model
from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model


def plot_detailed_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot detailed confusion matrix with percentages
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['Rotten', 'Fresh']
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Percentage matrix
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate confusion matrix')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--save_path', type=str, default='confusion_matrix.png',
                        help='Path to save confusion matrix')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    _, _, test_loader = get_loaders(data_dir=args.data_dir, batch_size=32)
    
    # Load model (assuming ResNet for now - can be made configurable)
    model = get_resnet_model(num_classes=1)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    _, y_true, y_pred, _ = evaluate_model(model, test_loader, device)
    
    # Plot confusion matrix
    plot_detailed_confusion_matrix(y_true, y_pred, save_path=args.save_path)


if __name__ == '__main__':
    main()



