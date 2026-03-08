"""
Visualization Module
Provides utilities for visualizing data, predictions, and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix


def plot_sample_images(dataloader, class_names, num_samples=8, save_path=None):
    """
    Plot sample images from the dataset
    
    Args:
        dataloader: DataLoader to sample from
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[idx].imshow(img.numpy())
        axes[idx].set_title(f'{class_names[labels[idx]]}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['Rotten', 'Fresh']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def visualize_predictions(model, dataloader, device, class_names, num_samples=8, save_path=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: PyTorch model
        dataloader: DataLoader to sample from
        device: Device to run inference on
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        probas = torch.sigmoid(outputs) if outputs.dim() == 1 else outputs
        preds = (probas > 0.5).long()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx].item()]
        confidence = probas[idx].item() if preds[idx] == 1 else 1 - probas[idx].item()
        
        axes[idx].imshow(img.numpy())
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})')
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



