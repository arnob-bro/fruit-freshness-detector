"""
VGG16 Transfer Learning Model
Uses pre-trained VGG16 for fruit freshness detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGTransfer(nn.Module):
    """VGG16-based transfer learning model"""

    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=True):
        super(VGGTransfer, self).__init__()

        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vgg16(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(num_classes=1, pretrained=True, freeze_backbone=True):
    return VGGTransfer(num_classes, pretrained, freeze_backbone)


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
