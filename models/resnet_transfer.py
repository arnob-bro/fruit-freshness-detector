"""
ResNet Transfer Learning Model
Uses pre-trained ResNet50 for fruit freshness detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetTransfer(nn.Module):
    """
    ResNet50-based transfer learning model
    """
    
    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=True):
        """
        Initialize ResNet transfer learning model
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone layers
        """
        super(ResNetTransfer, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(num_classes=1, pretrained=True, freeze_backbone=True):
    """
    Get ResNet transfer learning model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone layers
        
    Returns:
        ResNetTransfer model instance
    """
    return ResNetTransfer(num_classes, pretrained, freeze_backbone)


if __name__ == "__main__":
    # Test model
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")



