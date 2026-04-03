"""
Data Loader Module for Fruit Freshness Detection
Handles per-fruit data loading, preprocessing, and batch creation
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os


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
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loaders(data_dir='data', batch_size=32, num_workers=4, fruit=None):
    """
    Create data loaders for training and validation.

    Args:
        data_dir (str): Root directory containing data
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        fruit (str): If provided, loads data for a specific fruit (e.g., 'apple').
                     Expected structure: data_dir/<fruit>/train/, val/, test/
                     If None, loads from data_dir/train/, val/, test/ directly.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if fruit:
        base = os.path.join(data_dir, fruit)
    else:
        base = data_dir

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    train_data = datasets.ImageFolder(os.path.join(base, 'train'), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(base, 'val'), transform=val_transform)
    test_data = datasets.ImageFolder(os.path.join(base, 'test'), transform=val_transform)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, test_loader


def get_class_names(data_dir='data', fruit=None):
    """
    Get class names from the dataset.

    Args:
        data_dir (str): Root data directory
        fruit (str): Specific fruit to get classes for

    Returns:
        list: List of class names (e.g., ['fresh', 'rotten'])
    """
    if fruit:
        base = os.path.join(data_dir, fruit)
    else:
        base = data_dir
    train_data = datasets.ImageFolder(os.path.join(base, 'train'))
    return train_data.classes


def get_class_weights(data_dir='data', fruit=None):
    """
    Compute class weights for imbalanced datasets.

    Args:
        data_dir (str): Root data directory
        fruit (str): Specific fruit

    Returns:
        torch.Tensor: Class weights
    """
    if fruit:
        base = os.path.join(data_dir, fruit)
    else:
        base = data_dir

    train_data = datasets.ImageFolder(os.path.join(base, 'train'))
    targets = torch.tensor(train_data.targets)
    class_counts = torch.bincount(targets).float()
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    return weights
