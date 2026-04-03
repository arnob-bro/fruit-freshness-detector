"""
MobileNet Transfer Learning Model
Uses pre-trained MobileNetV2 for lightweight fruit freshness detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetTransfer(nn.Module):
    """
    MobileNetV2-based transfer learning model
    Lightweight model suitable for mobile deployment
    """
    
    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=True):
        """
        Initialize MobileNet transfer learning model
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone layers
        """
        super(MobileNetTransfer, self).__init__()
        
        # Load pre-trained MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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
    Get MobileNet transfer learning model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone layers
        
    Returns:
        MobileNetTransfer model instance
    """
    return MobileNetTransfer(num_classes, pretrained, freeze_backbone)


if __name__ == "__main__":
    # Test model
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")



