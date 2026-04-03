"""
Models package - 5 CNN architectures for fruit freshness classification
"""

from .mobilenet_transfer import get_model as get_mobilenet_model
from .resnet_transfer import get_model as get_resnet_model
from .efficientnet_transfer import get_model as get_efficientnet_model
from .vgg_transfer import get_model as get_vgg_model
from .densenet_transfer import get_model as get_densenet_model
from .cnn_scratch import get_model as get_cnn_model

# Architecture registry: maps config name -> (getter_fn, display_name)
ARCHITECTURES = {
    'mobilenet': (get_mobilenet_model, 'MobileNetV2'),
    'resnet': (get_resnet_model, 'ResNet50'),
    'efficientnet': (get_efficientnet_model, 'EfficientNetB0'),
    'vgg': (get_vgg_model, 'VGG16'),
    'densenet': (get_densenet_model, 'DenseNet121'),
}


def get_model_by_name(name, num_classes=1, pretrained=True, freeze_backbone=True):
    """Get a model by its architecture name."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown model: {name}. Choose from {list(ARCHITECTURES.keys())}")
    getter_fn, _ = ARCHITECTURES[name]
    return getter_fn(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


__all__ = [
    'get_cnn_model', 'get_resnet_model', 'get_mobilenet_model',
    'get_efficientnet_model', 'get_vgg_model', 'get_densenet_model',
    'get_model_by_name', 'ARCHITECTURES',
]
