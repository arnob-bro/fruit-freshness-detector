"""
Data Augmentation Module
Provides advanced augmentation techniques for fruit freshness detection
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import random


class AdvancedAugmentation:
    """Advanced augmentation techniques for fruit images"""
    
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])
    
    def __call__(self, img):
        return self.augmentations(img)


class MixUp:
    """MixUp augmentation technique"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        if random.random() > 0.5:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels


def get_augmentation_pipeline(mode='standard'):
    """
    Get augmentation pipeline based on mode
    
    Args:
        mode (str): 'standard', 'aggressive', or 'light'
        
    Returns:
        transforms.Compose: Augmentation pipeline
    """
    if mode == 'standard':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif mode == 'aggressive':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            AdvancedAugmentation(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # light
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



