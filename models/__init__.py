"""
Models package
"""

from .cnn_scratch import get_model as get_cnn_model
from .resnet_transfer import get_model as get_resnet_model
from .mobilenet_transfer import get_model as get_mobilenet_model

__all__ = ['get_cnn_model', 'get_resnet_model', 'get_mobilenet_model']



