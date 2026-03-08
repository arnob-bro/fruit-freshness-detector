"""
Data Loader Module for Fruit Freshness Detection
Handles data loading, preprocessing, and batch creation
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


def get_transforms(augment=True):
    """
    Get data transformation pipeline
    
    Args:
        augment (bool): Whether to apply data augmentation
        
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loaders(data_dir='data', batch_size=32, num_workers=4):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Root directory containing train/val/test folders
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    test_transform = get_transforms(augment=False)
    
    train_data = datasets.ImageFolder(f'{data_dir}/train', transform=train_transform)
    val_data = datasets.ImageFolder(f'{data_dir}/val', transform=val_transform)
    test_data = datasets.ImageFolder(f'{data_dir}/test', transform=test_transform)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir='data'):
    """
    Get class names from the dataset
    
    Args:
        data_dir (str): Root directory containing train/val/test folders
        
    Returns:
        list: List of class names
    """
    train_data = datasets.ImageFolder(f'{data_dir}/train')
    return train_data.classes



